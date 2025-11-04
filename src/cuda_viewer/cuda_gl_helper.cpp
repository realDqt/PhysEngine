#include "cuda_viewer/cuda_gl_helper.h"

void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
{
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,cudaGraphicsMapFlagsWriteDiscard));
}

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
}

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
    void *ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
                                                            *cuda_vbo_resource));
    return ptr;
}

void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}