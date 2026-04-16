#ifndef SKIPLIST_HPP
#define SKIPLIST_HPP

#include <vector>
#include <optional>
#include <functional>
#include <mutex>

/**
 * @brief Interfaz para una Cola de Prioridad Relajada.
 * @tparam T Tipo de dato (en Viltrum será ExtendedRegion).
 * @tparam Compare Comparador (equivalente a heap_ordering).
 */
template <typename T, typename Compare = std::less<T>>
class Skiplist {
public:
    // Alias para compatibilidad con el benchmark
    using value_type = T;

    /**
     * @brief Constructor que define el grado de relajación.
     * @param comp Instancia del comparador de prioridad.
     */
    explicit Skiplist(Compare comp = Compare()){
        this->n = n;
        this->comp = comp;
        this->current_level = 1;
        this->_size = 0;

        // Inicializamos NIL a nivel 0 ya que es el nodo final
        NIL = new node(T(), 0, nullptr);

        // Inicializamos el header a nivel máximo con todos sus punteros apuntando a NIL
        header = new node(T(), MAX_LEVEL, NIL);
    }

    /**
     * @brief Inserta un elemento de forma concurrente.
     */
    void push(const T& value) {
        node* update[MAX_LEVEL];
        node* curr = header;

        // Búsqueda inicial (sin locks) para rellenar update[]
        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            node* next = curr->forward[i].load(std::memory_order_acquire);
            while (next != NIL && comp(next->data, value)) {
                curr = next;
                next = curr->forward[i].load(std::memory_order_acquire);
            }
            update[i] = curr;
        }

        std::size_t lvl = random_level();
        node* newNode = new node(value, lvl, NIL);

        // Inserción de abajo hacia arriba con validación
        for (std::size_t i = 0; i < lvl; ++i) {
            bool level_inserted = false;
            node* predecessor = update[i];

            while (!level_inserted) {
                // Bloqueamos el predecesor que encontramos antes
                predecessor->node_lock.lock();

                // VALIDACIÓN: ¿Sigue siendo el nodo anterior correcto?
                node* nextNode = predecessor->forward[i].load(std::memory_order_acquire);
                
                // Si el siguiente nodo es menor que nuestro valor, alguien insertó algo en medio.
                if (nextNode != NIL && comp(nextNode->data, value)) {
                    // El predecesor ya no es válido para este nivel.
                    predecessor->node_lock.unlock();
                    
                    // Avanzamos: el nuevo predecesor es el nodo que se coló
                    predecessor = nextNode; 
                    
                    // Seguimos avanzando hasta encontrar el punto justo antes de 'value'
                    node* actualNext = predecessor->forward[i].load(std::memory_order_acquire);
                    while (actualNext != NIL && comp(actualNext->data, value)) {
                        predecessor = actualNext;
                        actualNext = predecessor->forward[i].load(std::memory_order_acquire);
                    }
                    // Reintentamos el bucle while con el nuevo predecessor
                } else {
                    // El lugar sigue siendo correcto. Insertamos newNode.
                    newNode->forward[i].store(nextNode, std::memory_order_relaxed);
                    predecessor->forward[i].store(newNode, std::memory_order_release);
                    
                    predecessor->node_lock.unlock();
                    level_inserted = true;
                }
            }
        }
        newNode->fully_linked.store(true, std::memory_order_release);
        _size.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Intenta extraer el elemento de mayor prioridad.
     * @return std::optional con el valor, o nullopt si la estructura está saturada o vacía.
     */
    std::optional<T> try_pop() {
        // Empezamos desde el primer nodo real
        node* curr = header->forward[0].load(std::memory_order_acquire);

        // Recorremos la lista hasta encontrar un nodo capturable o llegar al final
        while (curr != NIL) {
            // Solo intentamos capturar si el nodo está totalmente insertado y no está marcado
            if (curr->fully_linked.load(std::memory_order_acquire) && 
                !curr->marked_for_deletion.load(std::memory_order_relaxed)) {
                
                // Intento de captura atómica
                bool expected = false;
                if (curr->marked_for_deletion.compare_exchange_strong(expected, true)) {
                    T result = curr->data;
                    
                    // Sacamos el nodo de la estructura SkipList
                    detach_node(curr, result); 
                    
                    return result;
                }
            }
            
            // Si el nodo actual estaba ocupado, marcado o no listo, pasamos al siguiente inmediatamente sin esperar.
            curr = curr->forward[0].load(std::memory_order_acquire);
        }

        // Si llegamos aquí, hemos recorrido toda la lista y no hay nada disponible
        return std::nullopt;
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
        std::recursive_mutex node_lock;

        // Mientras el número aleatorio generado sea menor que P * base
        // y no hayamos alcanzado el límite máximo definido.
        // Usamos 100 como base para que el 0.5 de P sea fácil de comparar.
        while (fast_rand(100) < (P * 100) && lvl < MAX_LEVEL) {
            lvl++;
        }

        return lvl;
    }

    /**
     * @brief Función auxiliar usada para marcar y eliminar un nodo de la lista.
     */
    void detach_node(node* target, const T& value) {
        node* update[MAX_LEVEL];
        node* curr = header;

        // Buscamos los predecesores específicos para este nodo 'target'
        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            node* next = curr->forward[i].load(std::memory_order_acquire);
            while (next != NIL && (comp(next->data, value) || next == target)) {
                if (next == target) break;
                curr = next;
                next = curr->forward[i].load(std::memory_order_acquire);
            }
            update[i] = curr;
        }

        // Desenlazamos con locks de grano fino nivel a nivel
        for (int i = (int)target->forward.size() - 1; i >= 0; --i) {
            std::lock_guard<std::recursive_mutex> lock(update[i]->node_lock);
            if (update[i]->forward[i].load() == target) {
                update[i]->forward[i].store(target->forward[i].load(), std::memory_order_release);
            }
        }
        _size.fetch_sub(1, std::memory_order_relaxed);
    }

    static constexpr std::size_t MAX_LEVEL = 32; // Suficiente para miles de millones de elementos
    static constexpr float P = 0.5; // Probabilidad de subir de nivel
    node* header; // Nodo cabecera, es de nivel maximo y apunta al inicio de la estructura
    node* NIL; // Nodo centinela, marca el final de la estructura
    std::size_t current_level;
    Compare comp;
    std::atomic<std::size_t> _size{0}; // Numero de elementos en la estructura

    struct node {
        T data;
        std::vector<std::atomic<node*>> forward; // El puntero en la posicion i apunta al siguiente elemento de nivel i o superior
        std::atomic<bool> fully_linked{false}; // El nodo terminó su push
        std::atomic<bool> marked_for_deletion{false}; // El nodo está siendo extraído

        node(T val, std::size_t level, node* nil_ptr) // Da valor a data y reserva level espacios en foward inicializandolos a NIL
            : data(val), forward(level) { // Solo reservamos el tamaño
            
            // Inicializamos cada nivel manualmente
            for (std::size_t i = 0; i < level; ++i) {
                forward[i].store(nil_ptr, std::memory_order_relaxed);
            }
        }
    };
};

#endif