#ifndef SKIPLIST_HPP
#define SKIPLIST_HPP

#include <vector>
#include <optional>
#include <functional>
#include <mutex>
#include <atomic> // Para std::atomic, std::atomic_flag y std::memory_order_*
#include <thread> // Para std::this_thread::yield()
#include <array>  // Para std::array<...> en push y la estructura node
#include <random> // Para std::random_device en fast_rand
#include <immintrin.h> // Necesario para _mm_pause()

class Spinlock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
public:
    void lock() {
        int backoff = 1;
        while (locked.test_and_set(std::memory_order_acquire)) {
            // Exponential backoff: si fallamos, esperamos un poco más cada vez
            // usando la instrucción de hardware de pausa (muy rápida, no invoca al SO)
            for (int i = 0; i < backoff; ++i) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
                _mm_pause(); // Pausa la pipeline de la CPU sin saturar el bus
#else
                // Fallback para arquitecturas no x86 (como ARM)
                std::this_thread::yield(); 
#endif
            }
            if (backoff < 1024) {
                backoff *= 2; 
            } else {
                // Si ya esperamos mucho, entonces sí cedemos la CPU al SO
                std::this_thread::yield();
            }
        }
    }
    
    void unlock() {
        locked.clear(std::memory_order_release);
    }
};

/**
 * @brief Interfaz para una Cola de Prioridad Relajada.
 * @tparam T Tipo de dato (en Viltrum será ExtendedRegion).
 * @tparam Compare Comparador (equivalente a heap_ordering).
 */
template <typename T, typename Compare = std::greater<T>>
class Skiplist {
private:
    static constexpr std::size_t MAX_LEVEL = 32;

public:
    // Alias para compatibilidad con el benchmark
    using value_type = T;

    /**
     * @brief Constructor que define el grado de relajación.
     * @param comp Instancia del comparador de prioridad.
     */
    explicit Skiplist(Compare comp = Compare(), int rel_factor = 0){
        this->comp = comp;
        this->current_level = 1;
        this->_size = 0;
        this->relaxation_factor = rel_factor;

        NIL = new node(T(), 0, nullptr);
        header = new node(T(), MAX_LEVEL, NIL);
    }

    /*
    ~Skiplist() {
        node* curr = header;
        while (curr != nullptr) {
            node* next = nullptr;
            // Solo intentamos leer el siguiente si no somos el nodo final
            if (curr != NIL) {
                next = curr->forward[0].load(std::memory_order_relaxed);
            }
            delete curr;
            curr = next;
        }
    }
    */

    /**
     * @brief Inserta un elemento de forma concurrente.
     */
    void push(T value) {
        // 'update' guardará el predecesor de nuestro nuevo elemento en cada nivel.
        std::array<node*, MAX_LEVEL> update;
        node* pred = header;

        // PASO 1: Búsqueda relajada (WeakSearch)
        // Buscamos de arriba hacia abajo sin tomar cerrojos. 
        // Nota: Para evitar condiciones de carrera (UB) con un current_level no atómico,
        // iteramos desde el MAX_LEVEL. El costo es nulo porque las referencias a NIL se saltan al instante.
        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            node* next = pred->forward[i].load(std::memory_order_relaxed);
            
            // Avanzamos mientras el siguiente exista y deba ir antes en la cola de prioridad.
            // Si comp(next->data, value) es verdadero, next tiene MAYOR prioridad.
            while (next != NIL && comp(next->data, value)) {
                pred = next;
                next = pred->forward[i].load(std::memory_order_relaxed);
            }
            update[i] = pred;
        }

        // PASO 2: Creación del nodo
        // El nodo se crea y se asigna su nivel de forma local, por lo que es seguro 
        // porque aún no está enlazado a la lista general.
        std::size_t new_level = random_level();
        // Usamos nuestro asignador local libre de bloqueos
        node* new_node = allocate_node(value, new_level, NIL);

        // PASO 3: Inserción y validación Bottom-Up (de abajo hacia arriba)
        for (std::size_t i = 0; i < new_level; ++i) {
            pred = update[i];
            
            while (true) {
                pred->node_lock.lock();
                node* next = pred->forward[i].load(std::memory_order_relaxed);

                // Avanzamos si alguien insertó nodos delante de nosotros
                while (next != NIL && comp(next->data, value)) {
                    pred->node_lock.unlock();
                    pred = next;
                    pred->node_lock.lock();
                    next = pred->forward[i].load(std::memory_order_relaxed);
                }

                // VALIDACIÓN CRUCIAL: ¿Nos hemos anclado a un nodo que está siendo borrado?
                if (pred->marked_for_deletion.load(std::memory_order_relaxed)) {
                    pred->node_lock.unlock();
                    
                    // El predecesor es inválido. Reiniciamos la búsqueda exclusiva para este nivel
                    pred = header;
                    next = pred->forward[i].load(std::memory_order_relaxed);
                    while (next != NIL && comp(next->data, value)) {
                        pred = next;
                        next = pred->forward[i].load(std::memory_order_relaxed);
                    }
                    continue; // Volvemos al inicio del while(true) para intentar bloquear de nuevo
                }

                // Si pasamos la validación, el enlace es seguro
                new_node->forward[i].store(next, std::memory_order_relaxed);
                pred->forward[i].store(new_node, std::memory_order_release);
                pred->node_lock.unlock();
                
                break; // Terminamos la inserción en este nivel
            }
        }

        // PASO 4: Flags finales
        // Marcamos como fully_linked. Esto es especialmente útil para la operación de borrado o pop.
        new_node->fully_linked.store(true, std::memory_order_release);
        
        // Aumentamos el tamaño de manera atómica
        _size.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Intenta extraer el elemento de mayor prioridad (con relajación).
     * @return std::optional con el valor, o nullopt si la estructura está vacía.
     */
    /**
     * @brief Intenta extraer el elemento de mayor prioridad (con relajación).
     * @return std::optional con el valor, o nullopt si la estructura está vacía.
     */
    std::optional<T> try_pop() {
        node* target = nullptr;
        T extracted_value;

        // ====================================================================
        // FASE 1: Búsqueda Relajada y Marcado Lógico (Basado en Delete isGarbage)
        // ====================================================================
        node* curr = header->forward[0].load(std::memory_order_relaxed);
        
        // El factor de relajación define cuántos elementos "top" podemos saltar 
        // para evitar que todos los hilos colisionen en el nodo 0.
        int steps = fast_rand(relaxation_factor); 

        while (curr != NIL) {
            // Comprobamos si el nodo es un candidato válido sin tomar cerrojos aún
            if (!curr->marked_for_deletion.load(std::memory_order_relaxed) &&
                 curr->fully_linked.load(std::memory_order_relaxed)) {
                
                if (steps == 0) {
                    // Hemos alcanzado nuestro candidato. Tomamos el cerrojo (lock(y, level) del paper)
                    curr->node_lock.lock();
                    
                    // Doble comprobación post-cerrojo: ¿Lo marcó otro hilo justo antes?
                    if (!curr->marked_for_deletion.load(std::memory_order_relaxed) &&
                         curr->fully_linked.load(std::memory_order_relaxed)) {
                        
                        // ¡Nos lo adjudicamos! Lo marcamos como borrado lógico.
                        // Desde este momento, este nodo es "invisible" para otros pop().
                        curr->marked_for_deletion.store(true, std::memory_order_release);
                        extracted_value = curr->data;
                        target = curr;
                        curr->node_lock.unlock();
                        break; // Ya tenemos nuestro objetivo
                    }
                    // Si falló la comprobación, lo soltamos y seguimos buscando
                    curr->node_lock.unlock();
                } else {
                    steps--;
                }
            }
            curr = curr->forward[0].load(std::memory_order_relaxed);
            
            // Si nos pasamos de los elementos de la lista y todavía quedan "steps",
            // forzamos volver a empezar, pero esta vez cogiendo el primero que veamos.
            if (curr == NIL && steps > 0) {
                curr = header->forward[0].load(std::memory_order_relaxed);
                steps = 0; 
            }
        }

        // Si recorrimos todo y no había nada válido, la cola estaba vacía (o sus nodos tomados)
        if (target == nullptr) {
            return std::nullopt; 
        }

       // ====================================================================
        // FASE 2 y 3: Búsqueda de predecesores y desvinculación segura
        // ====================================================================
        node* pred = header;
        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            node* next = pred->forward[i].load(std::memory_order_relaxed);
            
            while (next != NIL && comp(next->data, extracted_value)) {
                pred = next;
                next = pred->forward[i].load(std::memory_order_relaxed);
            }
            
            while (next != NIL && next != target && !comp(extracted_value, next->data)) {
                pred = next;
                next = pred->forward[i].load(std::memory_order_relaxed);
            }

            if (next == target) {
                while (true) {
                    pred->node_lock.lock();
                    node* locked_next = pred->forward[i].load(std::memory_order_relaxed);
                    
                    // Avanzamos por si insertaron un nuevo nodo en medio
                    while (locked_next != target && locked_next != NIL) {
                        pred->node_lock.unlock();
                        pred = locked_next;
                        pred->node_lock.lock();
                        locked_next = pred->forward[i].load(std::memory_order_relaxed);
                    }
                    
                    // VALIDACIÓN CRUCIAL: ¿Es nuestro predecesor un nodo borrado por otro hilo?
                    if (pred->marked_for_deletion.load(std::memory_order_relaxed)) {
                        pred->node_lock.unlock();
                        
                        // Reiniciamos la búsqueda en este nivel
                        pred = header;
                        next = pred->forward[i].load(std::memory_order_relaxed);
                        while (next != NIL && comp(next->data, extracted_value)) {
                            pred = next;
                            next = pred->forward[i].load(std::memory_order_relaxed);
                        }
                        while (next != NIL && next != target && !comp(extracted_value, next->data)) {
                            pred = next;
                            next = pred->forward[i].load(std::memory_order_relaxed);
                        }
                        
                        // Si tras re-buscar el target ya no está en este nivel, salimos
                        if (next != target) break; 
                        
                        continue; // Reintentamos el bloqueo
                    }
                    
                    // Desvinculación física garantizada
                    if (locked_next == target) {
                        node* target_next = target->forward[i].load(std::memory_order_relaxed);
                        pred->forward[i].store(target_next, std::memory_order_release);
                    }
                    
                    pred->node_lock.unlock();
                    break; // Nivel completado
                }
            }
        }

        // ====================================================================
        // FASE 4: Limpieza final
        // ====================================================================
        _size.fetch_sub(1, std::memory_order_relaxed);
        
        // Lo mandamos al Garbage Collector local (que ya tienes definido en skiplist.hpp)
        retire_node(target); 
        
        return extracted_value;
    }

    /**
     * @brief Comprueba si la estructura esta vacia.
     */
    bool empty() const{
        return _size.load(std::memory_order_relaxed) == 0;
    }

    /**
     * @brief Retorna el número aproximado de elementos.
     */
    std::size_t size() const{
        return _size.load(std::memory_order_relaxed);
    }

private:
    struct node {
        T data;
        std::array<std::atomic<node*>, MAX_LEVEL> forward; 
        Spinlock node_lock;
        std::atomic<bool> fully_linked{false};
        std::atomic<bool> marked_for_deletion{false};

        node(T val, std::size_t level, node* nil_ptr) 
            : data(val) {
            for (std::size_t i = 0; i < MAX_LEVEL; ++i) {
                forward[i].store(nil_ptr, std::memory_order_relaxed);
            }
        }
    };

    /**
    * @brief Genera un numero aleatorio en [0, max_val]
    * @param max_val Valor máximo generado 
    */
    inline std::size_t fast_rand(std::size_t max_val) { // Algoritmo de numeros aleatorios Xorshift
        static thread_local uint32_t state = []() {
            uint32_t seed = std::random_device{}();
            return seed == 0 ? 1 : seed;
        }();
        
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        
        return state % max_val;
    }

    /**
     * @brief Genera un nivel aleatorio para un nuevo nodo.
     * Sigue la distribución P=0.5 descrita por William Pugh.
     */
    std::size_t random_level() {
        std::size_t lvl = 1;

        while (fast_rand(100) < (P * 100) && lvl < MAX_LEVEL) {
            lvl++;
        }

        return lvl;
    }

    static constexpr float P = 0.5; // Probabilidad de subir de nivel
    node* header; // Nodo cabecera, es de nivel maximo y apunta al inicio de la estructura
    node* NIL; // Nodo centinela, marca el final de la estructura
    std::size_t current_level;
    Compare comp;
    std::atomic<std::size_t> _size{0}; // Numero de elementos en la estructura
    int relaxation_factor;

    void retire_node(node* n) { // Actualmente no se borran
        static thread_local std::vector<node*> local_garbage;
        local_garbage.push_back(n);
    }

    node* allocate_node(T val, std::size_t level, node* nil_ptr) {
        // Cuántos nodos reservamos de golpe cada vez que hablamos con el SO
        constexpr int CHUNK_SIZE = 4096; 
        
        // Memoria en bruto. Al ser thread_local, cada hilo tiene su propia arena independiente
        static thread_local uint8_t* memory_block = nullptr;
        static thread_local int nodes_allocated = 0;

        // Si es la primera vez o nos hemos quedado sin espacio en nuestro bloque
        if (nodes_allocated == CHUNK_SIZE || memory_block == nullptr) {
            // Pedimos un gran bloque de memoria "cruda" al sistema operativo de una sola vez
            // ::operator new solo reserva el espacio (como malloc en C), sin llamar constructores
            memory_block = static_cast<uint8_t*>(::operator new(CHUNK_SIZE * sizeof(node)));
            nodes_allocated = 0;
        }

        // Calculamos la dirección de memoria exacta donde irá el nuevo nodo
        node* new_ptr = reinterpret_cast<node*>(memory_block + (nodes_allocated * sizeof(node)));
        
        // PLACEMENT NEW: Le decimos a C++ "Construye un objeto 'node' exactamente en esta dirección"
        new (new_ptr) node(val, level, nil_ptr);
        
        nodes_allocated++;
        
        return new_ptr;
    }
};

#endif