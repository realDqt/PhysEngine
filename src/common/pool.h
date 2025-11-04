#pragma once

#include <stdlib.h>

#include "math/math.h"
#include "common/logger.h"
#include "common/array.h"
#include "common/allocator.h"

PHYS_NAMESPACE_BEGIN

/**
 * @brief Pool class implements a basic object pool. 
 */
template<class T, size_t blockSize>
class Pool{
  public:
    Pool():m_usedNum(0),m_freeNum(0),m_freeNode(nullptr){
        m_alignAllocFunc = alignedAllocC11Std;
        m_alignFreeFunc = alignedFreeC11Std;
    }

    ~Pool(){
        //// destroy all T object
        destroyAllT();

        //// free all blocks
		for(auto block = m_blocks.begin(), blockEnd = m_blocks.end(); block != blockEnd; ++block)
			m_alignFreeFunc(*block, alignof(T));
    }

    /**
     * @brief create a new T object from the pool
     */
    template <class ...Args>
    FORCE_INLINE T* create(Args&&... args) {
		T* t = allocate();
		return t ? new (t) T(std::forward<Args>(args) ...) : nullptr;
	}

    /**
     * @brief destroy a T object, and return its space to the pool
     */
    FORCE_INLINE void destroy(T* const ptr) {
		if(ptr) {
			ptr->~T();
			deallocate(ptr);
		}
	}


    int getFreeNum(){ return m_freeNum; }//!< @brief Return the number of free nodes

    int getUsedNum(){ return m_usedNum; }//!< @brief Return the number of used nodes
    
    int getOccupiedSpace(){ return blockSize * m_blocks.size(); }//!< @brief Return the occupied space
    
  protected:
    
    ObjectArray<void*> m_blocks; //!< The memory blocks that the pool uses
    /**
     * @brief FreeNode structure is used to construct the free-node chain.
     */
  	struct FreeNode {
		FreeNode* next;
	};
    FreeNode* m_freeNode; //!< The header of the free-node chain
    int m_usedNum; //!< The number of used node
    int m_freeNum; //!< The number of free node

    void* (*m_alignAllocFunc)(size_t size, size_t align); //!< The memory alloc function
    void (*m_alignFreeFunc)(void* ptr, size_t align); //!< The memory release function
    
    /**
     * @brief Allocate a T object from the pool
     */
    FORCE_INLINE T* allocate() {
		if(m_freeNode == nullptr)
			allocBlock();
		T* ptr = reinterpret_cast<T*>(m_freeNode);
		m_freeNode = m_freeNode->next;
        m_usedNum++;
        m_freeNum--;
        return ptr;
	}
    
    /**
     * @brief Deallocate a T object
     */
    FORCE_INLINE void deallocate(T* ptr) {
		if(ptr) {
			ASSERT(m_usedNum);
            m_usedNum--;
			pushFreeNode(reinterpret_cast<FreeNode*>(ptr));
		}
	}

    /**
     * @brief Push the free-node into free-node chain
     */
    void pushFreeNode(FreeNode* ptr){
        if(ptr){
            m_freeNum++;
            ptr->next = m_freeNode;
            m_freeNode = ptr;
        }
    }

    /**
     * @brief Allocate a new block only when running out of the free-nodes
     */
    void allocBlock() {
		T* block = reinterpret_cast<T*>(m_alignAllocFunc(blockSize, alignof(T)));

		m_blocks.push_back((void*)block);

		T* it = block + (blockSize / sizeof(T));
		while(--it >= block)
			pushFreeNode(reinterpret_cast<FreeNode*>(it));
	}

    /**
     * @brief Release all T objects in destructor
     */
    void destroyAllT(){
        ObjectArray<T*> freeNodes;
        FreeNode* iter = m_freeNode;
        while(iter) {
			freeNodes.push_back(reinterpret_cast<T*> (iter));
			iter = iter->next;
		}
        
		static auto cmpTPtr = [=](T* i, T* j) { return i < j; };
		static auto cmpVPtr = [=](void* i, void* j) { return i < j; };
        std::sort(freeNodes.begin(), freeNodes.end(), cmpTPtr); //// low addr -> high addr
        std::sort(m_blocks.begin(), m_blocks.end(), cmpVPtr); //// low addr -> high addr

        int j = 0;
        for(int i = 0; i < m_blocks.size(); i++){
            T* elem = reinterpret_cast<T*> (m_blocks[i]);
            T* bEnd = elem + (blockSize / sizeof(T));
            for(; elem != bEnd; elem++){
                if(freeNodes[j] == elem)
                    j++;
                else if(freeNodes[j] > elem){
                    elem->~T();
                }else{
                    LOG_OSTREAM_ERROR << "freeNodes[j] < elem" << std::endl;
                }
            }
        }
    }
};

PHYS_NAMESPACE_END
