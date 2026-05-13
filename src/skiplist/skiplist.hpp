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
#include <immintrin.h>

// =============================================================================
// Spinlock (sin cambios)
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
    void unlock() {
        locked.clear(std::memory_order_release);
    }
};

// =============================================================================
// Skiplist
// Referencias:
//   [1] K. Fraser, "Practical lock-freedom", PhD thesis, Univ. of Cambridge,
//       2004. Cap. 2 (EBR) y Cap. 5 (memory management).
//       https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-579.pdf
//   [2] M. Michael, "Hazard Pointers: Safe Memory Reclamation for Lock-Free
//       Objects", IEEE TPDS, 2004. (slab allocator por thread, Sec. IV-B)
// =============================================================================
template <typename T, typename Compare = std::greater<T>>
class Skiplist {
private:
    static constexpr std::size_t MAX_LEVEL = 32;

    // =========================================================================
    // Nodo
    // =========================================================================
    struct node {
        // Almacenamiento alineado para T sin construirlo
        alignas(T) unsigned char data_buffer[sizeof(T)];
        
        std::array<std::atomic<node*>, MAX_LEVEL> forward;
        Spinlock node_lock;
        std::atomic<bool> fully_linked{false};
        std::atomic<bool> marked_for_deletion{false};

        // Constructor para nodos con DATOS reales
        node(T val, std::size_t /*level*/, node* nil_ptr) {
            new (data_buffer) T(std::move(val)); // Construcción manual
            for (std::size_t i = 0; i < MAX_LEVEL; ++i)
                forward[i].store(nil_ptr, std::memory_order_relaxed);
        }

        // NUEVO: Constructor para nodos CENTINELA (NIL y header)
        // No llama al constructor de T, dejando el buffer vacío.
        node(std::size_t /*level*/, node* nil_ptr) {
            for (std::size_t i = 0; i < MAX_LEVEL; ++i)
                forward[i].store(nil_ptr, std::memory_order_relaxed);
        }

        // Función auxiliar para acceder al dato de forma segura
        T& get_data() {
            return *reinterpret_cast<T*>(data_buffer);
        }
    };

    // =========================================================================
    // EBR: registro de épocas por thread
    // alignas(64) evita false sharing entre threads vecinos en el array
    // [1] Fraser 2004, Cap. 2.4
    // =========================================================================
    struct alignas(64) ThreadEpoch {
        std::atomic<uint64_t> value{UINT64_MAX}; // UINT64_MAX = thread inactivo
    };

    static constexpr int    MAX_THREADS     = 256;
    static constexpr int    CLEANUP_THRESHOLD = 64; // nodos retirados antes de intentar reclamar
    static constexpr uint64_t INACTIVE      = UINT64_MAX;

    // Época global compartida. Se incrementa periódicamente.
    alignas(64) std::atomic<uint64_t> global_epoch{0};

    // Tabla de épocas locales. Cada thread se registra con un índice único.
    std::array<ThreadEpoch, MAX_THREADS> thread_epochs{};

    // Contador atómico para asignar índices de thread
    std::atomic<int> next_thread_idx{0};

    // Índice de este thread en la tabla (thread_local por instancia no es posible
    // directamente, usamos un wrapper que se registra la primera vez)
    int get_thread_idx() {
        // thread_local dentro de una función miembro: cada thread tiene su propio
        // índice, asignado una sola vez por instancia de Skiplist.
        // Guardamos el puntero a la instancia para detectar si ya nos registramos.
        struct Registration {
            Skiplist* owner = nullptr;
            int idx = -1;
        };
        static thread_local Registration reg;
        if (reg.owner != this) {
            reg.owner = this;
            reg.idx = next_thread_idx.fetch_add(1, std::memory_order_relaxed);
            // Si superamos MAX_THREADS algo va muy mal; fallback al índice 0
            if (reg.idx >= MAX_THREADS) reg.idx = 0;
        }
        return reg.idx;
    }

    // =========================================================================
    // EBR: entrar/salir de sección crítica
    // =========================================================================
    void enter_epoch(int tidx) {
        // Publicamos la época global actual antes de tocar ningún puntero
        thread_epochs[tidx].value.store(
            global_epoch.load(std::memory_order_relaxed),
            std::memory_order_seq_cst);
    }

    void exit_epoch(int tidx) {
        thread_epochs[tidx].value.store(INACTIVE, std::memory_order_release);
    }

    // =========================================================================
    // Thread-local slab allocator
    // Cada thread tiene su propio bloque de memoria y su propia free-list.
    // No hay ningún lock en la ruta de asignación.
    // [1] Fraser 2004, Cap. 5.2 — [2] Michael 2004, Sec. IV-B
    // =========================================================================
    static constexpr int CHUNK_SIZE = 256; // nodos por chunk

    struct SlabChunk {
        uint8_t* memory;
        int      used;
        SlabChunk* next_chunk; // lista enlazada de chunks del thread

        SlabChunk() : used(0), next_chunk(nullptr) {
            memory = static_cast<uint8_t*>(::operator new(CHUNK_SIZE * sizeof(node)));
        }
        ~SlabChunk() { ::operator delete(memory); }
    };

    struct ThreadSlab {
        SlabChunk*  current = nullptr; // chunk activo
        node*       free_list = nullptr; // nodos reciclados listos para reusar

        node* allocate(T val, std::size_t level, node* nil_ptr) {
            // 1. Intentamos reusar un nodo reciclado de la free-list local
            if (free_list) {
                node* n = free_list;
                // El campo forward[0] lo usamos como puntero al siguiente libre
                free_list = free_list->forward[0].load(std::memory_order_relaxed);
                // Reconstruimos el nodo en el mismo espacio (placement new)
                n->~node();
                new (n) node(val, level, nil_ptr);
                return n;
            }

            // 2. Si no hay reciclados, tomamos del chunk activo
            if (!current || current->used == CHUNK_SIZE) {
                SlabChunk* new_chunk = new SlabChunk();
                new_chunk->next_chunk = current;
                current = new_chunk;
            }

            node* ptr = reinterpret_cast<node*>(
                current->memory + current->used * sizeof(node));
            new (ptr) node(val, level, nil_ptr);
            current->used++;
            return ptr;
        }

        // Devuelve un nodo a la free-list local (ya es seguro liberarlo)
        void recycle(node* n, node* nil_ptr) {
            n->get_data().~T(); // Destrucción explícita de T
            n->forward[0].store(free_list, std::memory_order_relaxed);
            free_list = n;
        }
    };

    // Cada thread tiene su propio slab. Como es thread_local y depende de la
    // instancia, lo indexamos igual que las épocas.
    // Usamos un array de punteros; cada thread crea su slab la primera vez.
    std::array<ThreadSlab*, MAX_THREADS> thread_slabs{};
    std::mutex slab_registry_mutex; // solo se toca al registrar un thread nuevo

    ThreadSlab& get_slab(int tidx) {
        if (!thread_slabs[tidx]) {
            std::lock_guard<std::mutex> lk(slab_registry_mutex);
            if (!thread_slabs[tidx])
                thread_slabs[tidx] = new ThreadSlab();
        }
        return *thread_slabs[tidx];
    }

    node* allocate_node(int tidx, T val, std::size_t level, node* nil_ptr) {
        return get_slab(tidx).allocate(val, level, nil_ptr);
    }

    // =========================================================================
    // EBR: basura local y reclamación
    // [1] Fraser 2004, Cap. 2.5
    // =========================================================================
    struct RetiredNode {
        node*    ptr;
        uint64_t epoch; // época en la que fue retirado
    };

    // Basura local por thread (igual que el slab, indexada por tidx)
    struct ThreadGarbage {
        std::vector<RetiredNode> list;
    };
    std::array<ThreadGarbage*, MAX_THREADS> thread_garbage{};
    std::mutex garbage_registry_mutex;

    ThreadGarbage& get_garbage(int tidx) {
        if (!thread_garbage[tidx]) {
            std::lock_guard<std::mutex> lk(garbage_registry_mutex);
            if (!thread_garbage[tidx])
                thread_garbage[tidx] = new ThreadGarbage();
        }
        return *thread_garbage[tidx];
    }

    void retire_node(int tidx, node* n) {
        uint64_t epoch = global_epoch.load(std::memory_order_relaxed);
        auto& gc = get_garbage(tidx);
        gc.list.push_back({n, epoch});

        if ((int)gc.list.size() >= CLEANUP_THRESHOLD)
            try_reclaim(tidx);
    }

    void try_reclaim(int tidx) {
        // Calculamos la época mínima activa entre todos los threads registrados
        // [1] Fraser 2004, Cap. 2.4 — ningún thread activo está por debajo de min_epoch
        uint64_t min_epoch = INACTIVE;
        int registered = next_thread_idx.load(std::memory_order_relaxed);
        for (int i = 0; i < registered && i < MAX_THREADS; ++i) {
            uint64_t e = thread_epochs[i].value.load(std::memory_order_acquire);
            if (e < min_epoch) min_epoch = e;
        }

        // Liberamos los nodos retirados antes de min_epoch: ningún thread activo
        // puede tener un puntero a ellos
        auto& gc = get_garbage(tidx);
        auto& slab = get_slab(tidx);
        auto& v = gc.list;

        v.erase(std::remove_if(v.begin(), v.end(),
            [&](const RetiredNode& rn) {
                if (rn.epoch < min_epoch) {
                    slab.recycle(rn.ptr, NIL);
                    return true;
                }
                return false;
            }), v.end());

        // Avanzamos la época global para que el progreso continúe
        global_epoch.fetch_add(1, std::memory_order_relaxed);
    }

    // =========================================================================
    // Generador de nivel aleatorio (Xorshift, sin cambios)
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
    // Estado de la skiplist
    // =========================================================================
    node*                    header;
    node*                    NIL;
    std::size_t              current_level;
    Compare                  comp;
    std::atomic<std::size_t> _size{0};
    int                      relaxation_factor;

public:
    // =========================================================================
    // Interfaz pública — idéntica a la versión original
    // =========================================================================
    using value_type = T;

    explicit Skiplist(Compare comp = Compare(), int rel_factor = 0)
        : comp(comp), current_level(1), relaxation_factor(rel_factor)
    {
        thread_slabs.fill(nullptr);
        thread_garbage.fill(nullptr);

        // NIL y header se crean directamente con new (son singleton de la lista)
        NIL    = new node(0, nullptr); 
        header = new node(MAX_LEVEL, NIL);
    }

    ~Skiplist() {
        // Liberamos toda la basura pendiente y los slabs de cada thread
        for (int i = 0; i < MAX_THREADS; ++i) {
            if (thread_garbage[i]) {
                delete thread_garbage[i];
                thread_garbage[i] = nullptr;
            }
            if (thread_slabs[i]) {
                // Los SlabChunks tienen su propio destructor que libera la memoria
                SlabChunk* chunk = thread_slabs[i]->current;
                while (chunk) {
                    SlabChunk* next = chunk->next_chunk;
                    delete chunk;
                    chunk = next;
                }
                delete thread_slabs[i];
                thread_slabs[i] = nullptr;
            }
        }
        delete header;
        delete NIL;
    }

    // -------------------------------------------------------------------------
    void push(T value) {
        int tidx = get_thread_idx();
        enter_epoch(tidx);

        std::array<node*, MAX_LEVEL> update;
        node* pred = header;

        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            node* next = pred->forward[i].load(std::memory_order_relaxed);
            while (next != NIL && comp(next->get_data(), value)) {
                pred = next;
                next = pred->forward[i].load(std::memory_order_relaxed);
            }
            update[i] = pred;
        }

        std::size_t new_level = random_level();
        node* new_node = allocate_node(tidx, value, new_level, NIL);

        for (std::size_t i = 0; i < new_level; ++i) {
            pred = update[i];
            while (true) {
                pred->node_lock.lock();
                node* next = pred->forward[i].load(std::memory_order_relaxed);

                while (next != NIL && comp(next->get_data(), value)) {
                    pred->node_lock.unlock();
                    pred = next;
                    pred->node_lock.lock();
                    next = pred->forward[i].load(std::memory_order_relaxed);
                }

                if (pred->marked_for_deletion.load(std::memory_order_relaxed)) {
                    pred->node_lock.unlock();
                    pred = header;
                    next = pred->forward[i].load(std::memory_order_relaxed);
                    while (next != NIL && comp(next->get_data(), value)) {
                        pred = next;
                        next = pred->forward[i].load(std::memory_order_relaxed);
                    }
                    continue;
                }

                new_node->forward[i].store(next, std::memory_order_relaxed);
                pred->forward[i].store(new_node, std::memory_order_release);
                pred->node_lock.unlock();
                break;
            }
        }

        new_node->fully_linked.store(true, std::memory_order_release);
        _size.fetch_add(1, std::memory_order_relaxed);

        exit_epoch(tidx);
    }

    // -------------------------------------------------------------------------
    std::optional<T> try_pop() {
        int tidx = get_thread_idx();
        enter_epoch(tidx);

        node* target = nullptr;
        T extracted_value;

        // En skiplist.hpp, dentro de try_pop()
        std::size_t current_size = _size.load(std::memory_order_relaxed);
        int steps = (relaxation_factor > 0 && current_size > (std::size_t)relaxation_factor) 
                    ? (int)fast_rand(relaxation_factor) : 0;

        // FASE 1: búsqueda relajada y marcado lógico
        node* curr = header->forward[0].load(std::memory_order_relaxed);

        while (curr != NIL) {
            if (!curr->marked_for_deletion.load(std::memory_order_relaxed) &&
                 curr->fully_linked.load(std::memory_order_relaxed)) {

                if (steps == 0) {
                    curr->node_lock.lock();
                    if (!curr->marked_for_deletion.load(std::memory_order_relaxed) &&
                         curr->fully_linked.load(std::memory_order_relaxed)) {
                        curr->marked_for_deletion.store(true, std::memory_order_release);
                        extracted_value = curr->get_data();
                        target = curr;
                        curr->node_lock.unlock();
                        break;
                    }
                    curr->node_lock.unlock();
                } else {
                    steps--;
                }
            }
            curr = curr->forward[0].load(std::memory_order_relaxed);

            if (curr == NIL && steps > 0) {
                curr = header->forward[0].load(std::memory_order_relaxed);
                steps = 0;
            }
        }

        if (target == nullptr) {
            exit_epoch(tidx);
            return std::nullopt;
        }

        // FASE 2 y 3: desvinculación física
        node* pred = header;
        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            node* next = pred->forward[i].load(std::memory_order_relaxed);

            while (next != NIL && comp(next->get_data(), extracted_value))  {
                pred = next;
                next = pred->forward[i].load(std::memory_order_relaxed);
            }
            while (next != NIL && next != target && !comp(extracted_value, next->get_data())) {
                pred = next;
                next = pred->forward[i].load(std::memory_order_relaxed);
            }

            if (next == target) {
                while (true) {
                    pred->node_lock.lock();
                    node* locked_next = pred->forward[i].load(std::memory_order_relaxed);

                    while (locked_next != target && locked_next != NIL) {
                        pred->node_lock.unlock();
                        pred = locked_next;
                        pred->node_lock.lock();
                        locked_next = pred->forward[i].load(std::memory_order_relaxed);
                    }

                    if (pred->marked_for_deletion.load(std::memory_order_relaxed)) {
                        pred->node_lock.unlock();
                        pred = header;
                        next = pred->forward[i].load(std::memory_order_relaxed);
                        while (next != NIL && comp(next->get_data(), extracted_value)) {
                            pred = next;
                            next = pred->forward[i].load(std::memory_order_relaxed);
                        }
                        while (next != NIL && next != target && !comp(extracted_value, next->get_data())) {
                            pred = next;
                            next = pred->forward[i].load(std::memory_order_relaxed);
                        }
                        if (next != target) break;
                        continue;
                    }

                    if (locked_next == target) {
                        node* target_next = target->forward[i].load(std::memory_order_relaxed);
                        pred->forward[i].store(target_next, std::memory_order_release);
                    }
                    pred->node_lock.unlock();
                    break;
                }
            }
        }

        // FASE 4: retiro seguro vía EBR en lugar de acumular sin liberar
        _size.fetch_sub(1, std::memory_order_relaxed);
        retire_node(tidx, target);

        exit_epoch(tidx);
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
    // drain(): extrae todos los elementos restantes de forma segura.
    // Para uso single-threaded al final del procesamiento.
    // No pasa por try_pop() para evitar el problema de relajación con pocos elementos.
    std::vector<T> drain() {
        std::vector<T> result;
        int tidx = get_thread_idx();
        enter_epoch(tidx);

        node* curr = header->forward[0].load(std::memory_order_relaxed);
        while (curr != NIL) {
            if (!curr->marked_for_deletion.load(std::memory_order_relaxed) &&
                 curr->fully_linked.load(std::memory_order_relaxed)) {
                result.push_back(curr->get_data());
                curr->marked_for_deletion.store(true, std::memory_order_relaxed);
                _size.fetch_sub(1, std::memory_order_relaxed);
                node* next = curr->forward[0].load(std::memory_order_relaxed);
                retire_node(tidx, curr);
                curr = next;
            } else {
                curr = curr->forward[0].load(std::memory_order_relaxed);
            }
        }

        exit_epoch(tidx);
        try_reclaim(tidx); // forzamos limpieza final
        return result;
    }
};

#endif