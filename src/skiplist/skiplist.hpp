#ifndef SKIPLIST_HPP
#define SKIPLIST_HPP

#include <vector>
#include <optional>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>
#include <array>
#include <random>
#include <algorithm>
#include <immintrin.h>

// =============================================================================
// Spinlock
// =============================================================================
class Spinlock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
public:
    void lock() {
        int backoff = 1;
        while (locked.test_and_set(std::memory_order_acquire)) {
            for (int i = 0; i < backoff; ++i) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            }
            if (backoff < 1024) backoff *= 2;
            else std::this_thread::yield();
        }
    }

    bool try_lock() {
        return !locked.test_and_set(std::memory_order_acquire);
    }

    void unlock() {
        locked.clear(std::memory_order_release);
    }
};

// =============================================================================
// Skiplist
//
// Gestión de memoria simplificada respecto a la versión original:
//   - Se eliminan el slab allocator por thread y el EBR (Epoch-Based Reclamation)
//     porque su interacción con OpenMP generaba use-after-free y corrupción de
//     punteros bajo alta contención.
//   - Los nodos se asignan con `new` y se liberan de forma diferida: se acumulan
//     en `retired_nodes` (protegido por `retired_mutex`) y se destruyen en el
//     destructor de la lista, cuando ya no hay ningún hilo activo.
//   - El resto de la lógica (inserción con locking por dirección de memoria,
//     extracción lazy, relajación) se mantiene igual.
// =============================================================================
template <typename T, typename Compare = std::greater<T>>
class Skiplist {
private:
    static constexpr std::size_t MAX_LEVEL = 32;

    // =========================================================================
    // Nodo
    // =========================================================================
    struct node {
        alignas(T) unsigned char data_buffer[sizeof(T)];

        std::array<std::atomic<node*>, MAX_LEVEL> forward;
        Spinlock node_lock;
        std::atomic<bool> fully_linked{false};
        std::atomic<bool> marked_for_deletion{false};

        // Constructor para nodos con datos reales
        node(T val, std::size_t /*level*/, node* nil_ptr) {
            new (data_buffer) T(std::move(val));
            for (std::size_t i = 0; i < MAX_LEVEL; ++i)
                forward[i].store(nil_ptr, std::memory_order_relaxed);
        }

        // Constructor para nodos centinela (NIL y header) — no construye T
        node(std::size_t /*level*/, node* nil_ptr) {
            for (std::size_t i = 0; i < MAX_LEVEL; ++i)
                forward[i].store(nil_ptr, std::memory_order_relaxed);
        }

        T& get_data() {
            return *reinterpret_cast<T*>(data_buffer);
        }
    };

    // =========================================================================
    // Generador de nivel aleatorio (Xorshift)
    // =========================================================================
    static constexpr float P = 0.5f;

    inline std::size_t fast_rand(std::size_t max_val) {
        static thread_local uint32_t state = []() {
            uint32_t seed = std::random_device{}();
            return seed == 0 ? 1 : seed;
        }();
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state % max_val;
    }

    std::size_t random_level() {
        std::size_t lvl = 1;
        while (fast_rand(100) < (std::size_t)(P * 100) && lvl < MAX_LEVEL)
            lvl++;
        return lvl;
    }

    // =========================================================================
    // Estado
    // =========================================================================
    node*                    header;
    node*                    NIL;
    std::size_t              current_level;
    Compare                  comp;
    std::atomic<std::size_t> _size{0};
    int                      relaxation_factor;

    // Lista de nodos retirados — se destruyen en el destructor
    std::mutex           retired_mutex;
    std::vector<node*>   retired_nodes;

    void retire_node(node* n) {
        std::lock_guard<std::mutex> lk(retired_mutex);
        retired_nodes.push_back(n);
    }

public:
    // =========================================================================
    // Interfaz pública
    // =========================================================================
    using value_type = T;

    explicit Skiplist(Compare comp = Compare(), int rel_factor = 0)
        : current_level(1), comp(comp), relaxation_factor(rel_factor)
    {
        NIL    = new node(std::size_t(0), nullptr);
        header = new node(MAX_LEVEL, NIL);
    }

    ~Skiplist() {
        // Destruir nodos vivos que quedaron en la lista sin ser extraídos
        node* curr = header->forward[0].load(std::memory_order_relaxed);
        while (curr != NIL) {
            node* next = curr->forward[0].load(std::memory_order_relaxed);
            
            // SOLO destruimos si NO ha sido retirado ya por try_pop() o drain()
            if (!curr->marked_for_deletion.load(std::memory_order_relaxed)) {
                curr->get_data().~T();
                delete curr;
            }
            curr = next;
        }
        delete header;
        delete NIL;

        // Destruir nodos retirados de forma segura (una sola vez)
        for (node* n : retired_nodes)
            delete n;
    }

    // -------------------------------------------------------------------------
    void push(T value) {
        std::size_t new_level = random_level();
        node* new_node = new node(value, new_level, NIL);

        std::array<node*, MAX_LEVEL> preds;
        std::array<node*, MAX_LEVEL> succs;

        while (true) {
            // Búsqueda de predecesores (sin lock)
            node* curr = header;
            for (int i = MAX_LEVEL - 1; i >= 0; --i) {
                node* next = curr->forward[i].load(std::memory_order_acquire);
                while (next != NIL &&
                       !next->marked_for_deletion.load(std::memory_order_acquire) &&
                       comp(next->get_data(), value)) {
                    curr = next;
                    next = curr->forward[i].load(std::memory_order_acquire);
                }
                preds[i] = curr;
                succs[i] = next;
            }

            // Recopilar predecesores únicos y ordenarlos por dirección de
            // memoria para garantizar un orden total global de adquisición
            // de locks y evitar deadlock entre hilos concurrentes.
            std::vector<node*> to_lock;
            to_lock.reserve(new_level);
            for (std::size_t i = 0; i < new_level; ++i) {
                if (std::find(to_lock.begin(), to_lock.end(), preds[i]) == to_lock.end())
                    to_lock.push_back(preds[i]);
            }
            std::sort(to_lock.begin(), to_lock.end());
            for (node* n : to_lock) n->node_lock.lock();

            // Validar que todos los predecesores siguen siendo válidos
            bool failed = false;
            for (std::size_t i = 0; i < new_level; ++i) {
                if (preds[i]->marked_for_deletion.load(std::memory_order_acquire) ||
                    preds[i]->forward[i].load(std::memory_order_acquire) != succs[i]) {
                    failed = true;
                    break;
                }
            }

            if (failed) {
                for (node* n : to_lock) n->node_lock.unlock();
                continue;
            }

            // Enlace físico protegido en todos los niveles
            for (std::size_t i = 0; i < new_level; ++i) {
                new_node->forward[i].store(succs[i], std::memory_order_relaxed);
                preds[i]->forward[i].store(new_node, std::memory_order_release);
            }

            new_node->fully_linked.store(true, std::memory_order_release);
            for (node* n : to_lock) n->node_lock.unlock();
            break;
        }

        _size.fetch_add(1, std::memory_order_relaxed);
    }

    // -------------------------------------------------------------------------
    std::optional<T> try_pop() {
        node* target = nullptr;
        T extracted_value;

        // Pasos de relajación
        std::size_t current_size = _size.load(std::memory_order_relaxed);
        int steps = (relaxation_factor > 0 && current_size > (std::size_t)relaxation_factor)
                    ? (int)fast_rand(relaxation_factor) : 0;

        // FASE 1: Búsqueda lógica — marcar el nodo candidato
        node* curr = header->forward[0].load(std::memory_order_acquire);
        while (curr != NIL) {
            if (curr->fully_linked.load(std::memory_order_acquire) &&
                !curr->marked_for_deletion.load(std::memory_order_acquire)) {

                if (steps <= 0) {
                    if (curr->node_lock.try_lock()) {
                        if (!curr->marked_for_deletion.load(std::memory_order_relaxed)) {
                            curr->marked_for_deletion.store(true, std::memory_order_release);
                            extracted_value = curr->get_data();
                            target = curr;
                            curr->node_lock.unlock();
                            break;
                        }
                        curr->node_lock.unlock();
                    }
                } else {
                    steps--;
                }
            }
            curr = curr->forward[0].load(std::memory_order_acquire);
        }

        if (target == nullptr)
            return std::nullopt;

        // FASE 2: Desvinculación física — un nivel a la vez, pred resetea a header
        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            bool unlinked = false;
            
            // Bucle de reintento obligatorio
            while (!unlinked) {
                node* pred = header;
                node* next = pred->forward[i].load(std::memory_order_acquire);

                while (next != NIL && next != target) {
                    // Saltamos nodos marcados para no quedarnos atascados
                    if (next->marked_for_deletion.load(std::memory_order_acquire) ||
                        comp(next->get_data(), extracted_value) ||
                        !comp(extracted_value, next->get_data())) {
                        pred = next;
                        next = pred->forward[i].load(std::memory_order_acquire);
                    } else {
                        break;
                    }
                }

                if (next == target) {
                    pred->node_lock.lock();
                    
                    // CORRECCIÓN: Validar que pred no solo apunte a target, sino que siga vivo
                    if (pred->forward[i].load(std::memory_order_relaxed) == target &&
                        !pred->marked_for_deletion.load(std::memory_order_relaxed)) { // <-- NUEVO
                        
                        node* target_next = target->forward[i].load(std::memory_order_relaxed);
                        pred->forward[i].store(target_next, std::memory_order_release);
                        unlinked = true; // Éxito
                    }
                    pred->node_lock.unlock();
                    
                    // Si la validación falla (ej. pred fue marcado), unlinked sigue siendo 'false' 
                    // y el bucle while(!unlinked) te obliga a reiniciar desde el header correctamente.
                } else {
                    // No se encontró en este nivel. Puede que el nodo original
                    // no haya llegado a crecer hasta este nivel (new_level).
                    unlinked = true; 
                }
            }
        }

        // FASE 3: Destruir el dato y programar la liberación del nodo
        _size.fetch_sub(1, std::memory_order_relaxed);
        target->get_data().~T();  // destrucción explícita antes de retirar
        retire_node(target);

        return extracted_value;
    }

    // -------------------------------------------------------------------------
    bool empty() const {
        return _size.load(std::memory_order_relaxed) == 0;
    }

    std::size_t size() const {
        return _size.load(std::memory_order_relaxed);
    }

    // -------------------------------------------------------------------------
    // drain(): extrae todos los elementos restantes.
    // Para uso single-threaded al final del procesamiento paralelo.
    std::vector<T> drain() {
        std::vector<T> result;

        node* curr = header->forward[0].load(std::memory_order_relaxed);
        while (curr != NIL) {
            node* next = curr->forward[0].load(std::memory_order_relaxed);
            if (!curr->marked_for_deletion.load(std::memory_order_relaxed) &&
                 curr->fully_linked.load(std::memory_order_relaxed)) {
                result.push_back(curr->get_data());
                curr->marked_for_deletion.store(true, std::memory_order_relaxed);
                _size.fetch_sub(1, std::memory_order_relaxed);
                curr->get_data().~T();
                retire_node(curr);
            }
            curr = next;
        }

        return result;
    }
};

#endif
