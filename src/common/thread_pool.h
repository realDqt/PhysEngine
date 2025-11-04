#pragma once
#include <taskflow/taskflow.hpp>

#include "common/general.h"
#include "common/array.h"
#include "common/timer.h"

PHYS_NAMESPACE_BEGIN

// TODO
// 1. provide more interface: depend on what we need.

/**
 * @brief ThreadPool class provide some useful interfaces based on TaskFlow. 
 */
class ThreadPool {
  public:
    // static tf::Taskflow taskflow;
    static tf::Executor executor;

    /**
     * @brief Provide the basic interface for loop-based parallel
     */
    template <typename F>
    static void parallelFor(int start, int end, const F &func) {
        
        tf::Taskflow taskflow;
        size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
        size_t g = std::max((end - start + w - 1) / w, size_t{1});
        
        taskflow.parallel_for(start, end, 1, func, g);
        executor.run(taskflow).get();
        // taskflow.dump(std::cout);
    }

    template <typename F>
    static void parallelForWithThreads(int start, int end, unsigned int workers, const F &func) {
        
        tf::Taskflow taskflow;
        size_t w = std::max(unsigned{1}, workers);
        size_t g = std::max((end - start + w - 1) / w, size_t{2});
      
        taskflow.parallel_for(start, end, 1, func, g);
        
        executor.run(taskflow).get();
        // taskflow.dump(std::cout);
    }

    /**
     * @brief Provide an interface for loop-based parallel where the result of each iterator are stacked into the result_array.
     * @param beg is the begin index
     * @param end is the end index
     * @param result_array is the array of objects T from add_to_array_op
     * @param add_to_array_op (ObjectArray<T>& res_arr, I i) is the function runing on the index i and push the result object back to the res_arr.
     */
    template <typename I, typename T, typename G>
    static void reduceToArray(I beg, I end, ObjectArray<T>& result_array, G&& add_to_array_op) {
        tf::Taskflow taskflow;
        size_t d = end-beg;
        //// workers
        size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
        //// group size
        size_t g = std::max((d + w - 1) / w, size_t{2});
        
        auto g_arrays = new ObjectArray<T>[w];
        reduceToArrayInternal(taskflow, beg, end, g_arrays, result_array, add_to_array_op);

        executor.run(taskflow).get();
        executor.wait_for_all();
        
        delete [] g_arrays;
    }

  protected:
    template <typename I, typename T, typename G>
    static std::pair<tf::Task, tf::Task> reduceToArrayInternal(tf::Taskflow& taskflow, I beg, I end, ObjectArray<T>* g_arrays, ObjectArray<T>& result_array, G&& add_to_array_op) {
            
      size_t d = end-beg;

      auto source = taskflow.placeholder();
      auto target = taskflow.placeholder();

      //// workers
      size_t w = std::max(unsigned{1}, std::thread::hardware_concurrency());
      //// group size
      size_t g = std::max((d + w - 1) / w, size_t{2});

      
      size_t id = 0;
      size_t remain = d;

      //// map
      while(beg != end) {

        auto e = beg;
        
        size_t x = std::min(remain, g);
        e += x;
        remain -= x;
        
        //// create a task for indices between [beg, e)
        auto task = taskflow.emplace([beg, e, add_to_array_op, res_arr = &g_arrays[id]] () mutable {
          I i = beg;
          for(; i != e; ++i) {
            // res_arr->emplace_back(add_to_array_op(i));
            add_to_array_op(res_arr, i);
          }
        });
        source.precede(task);
        task.precede(target);

        beg = e;
        id ++;
      }

      //// reduce
      target.work([w=id, g_arrays, &result_array] () {
        for(auto i=0u; i<w; i++) {
          result_array.insert(result_array.end(), g_arrays[i].begin(), g_arrays[i].end());
        }
      });

      return std::make_pair(source, target); 
    }


};

PHYS_NAMESPACE_END