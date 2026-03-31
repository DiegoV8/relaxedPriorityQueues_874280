#ifndef MULTI_QUEUE_HPP
#define MULTI_QUEUE_HPP

#include <vector>
#include <optional>
#include <functional>
#include <random>
#include <mutex>

/**
 * @brief Interfaz para una Multiqueue.
 * @tparam T Tipo de dato.
 * @tparam Compare Comparador.
 */
template <typename T, typename Compare = std::less<T>>
class Multiqueue {
public:
    // Alias para compatibilidad con los tests de estructuras de datos
    using value_type = T;

    /**
     * @brief Constructor que define el grado de relajación.
     * @param n Número de sub-colas internas.
     * @param c Número de sub-colas que se comparan al extraer elementos.
     * @param comp Instancia del comparador de prioridad.
     */
    explicit Multiqueue(std::size_t n = 8, int c = 2, Compare comp = Compare()) {
        this->n = n;
        this->c = c;
        this->comp = comp;

        // Reservamos espacio para evitar reasignaciones
        queues.reserve(n);
        
        // Inicializamos cada shard individualmente
        for (std::size_t i = 0; i < n; ++i) {
            queues.emplace_back(comp);
        }
    }

    /**
     * @brief Inserta un elemento de forma concurrente.
     */
    void push(const T& value){
        // Cada thread tiene un generador de numeros aleatorios distinto
        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<std::size_t> distribution(0, n - 1);
        
        // Insercion
        std::size_t i = distribution(generator);
        {
            std::lock_guard<std::mutex> lock(queues[i].M); // Bloqueamos la cola antes de insertar nada, se desbloquea al destruirse lock
            queues[i].pq.push(value);
        }
    }

    /**
     * @brief Intenta extraer el elemento de mayor prioridad.
     * @return std::optional con el valor, o nullopt si la estructura está saturada o vacía.
     */
    std::optional<T> try_pop() {
        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<std::size_t> distribution(0, n - 1);

        // Guardaremos los índices de las colas que logremos bloquear
        std::vector<std::size_t> locked_indices;
        int best_idx = -1;

        // Intentamos muestrear y bloquear hasta 'c' colas
        for (int i = 0; i < c; ++i) {
            std::size_t idx = distribution(generator);
            
            // Intentamos bloquear sin quedarnos dormidos (try_lock)
            if (queues[idx].M.try_lock()) {
                if (!queues[idx].pq.empty()) {
                    locked_indices.push_back(idx);
                    
                    // Comprobamos si es el elemento con mayor prioridad que hemos encontrado
                    if (best_idx == -1 || comp(queues[best_idx].pq.top(), queues[idx].pq.top())) {
                        best_idx = idx;
                    }
                } else {
                    // Si estaba vacía, la desbloqueamos ya
                    queues[idx].M.unlock();
                }
            }
        }

        // Si no hemos podido bloquear ninguna con datos, devolvemos vacío
        if (best_idx == -1) {
            for (std::size_t idx : locked_indices) queues[idx].M.unlock();
            return std::nullopt;
        }

        // Obtenemos el mejor valor
        T result = queues[best_idx].pq.top();
        queues[best_idx].pq.pop();

        // Desbloqueamos el resto de colas
        for (std::size_t idx : locked_indices) {
            queues[idx].M.unlock();
        }

        return result;
    }

    /**
     * @brief Comprueba si todas las colas internas están vacías.
     */
    bool empty() const {
        for (std::size_t i = 0; i < n; i++){
            std::lock_guard<std::mutex> lock(queues[i].M); // Bloqueamos el mutex para que no se actualice mientras comprobamos el resto
            if (!queues[i].pq.empty()) return false;
        }
        return true;
    }

    /**
     * @brief Retorna el número aproximado de elementos.
     */
    std::size_t size() const {
        std::size_t sum = 0;
        for (std::size_t i = 0; i < n; i++){
            std::lock_guard<std::mutex> lock(queues[i].M); // Bloqueamos el mutex para que no se actualice mientras sumamos el resto
            sum += queues[i].pq.size();
        }
        return sum;
    }

private:
    struct Q { // Representa cada una de las colas de la multiqueue
        std::priority_queue<T, std::vector<T>, Compare> pq;
        std::mutex M;

        // Inicializa la pq con el comparador
        Q(const Compare& comp) : pq(comp) {}
    };

    std::vector<Q> queues; // Vector de Colas
    std::size_t n; // Numero de colas en la Multiqueue
    int c; // Numero de colas a comparar durante la extracción, típicamente 2
    Compare comp; // Comparador de prioridad de elementos
    
};

#endif