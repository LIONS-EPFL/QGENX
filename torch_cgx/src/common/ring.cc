#include "ring.h"

namespace cgx {
namespace common {

MPI_Allreduce_Ring::MPI_Allreduce_Ring(GPUContext *gpu_context,
                                       std::shared_ptr<Compressor> compressor,
                                       std::shared_ptr<Communicator> communicator,
                                       int world_size) : MPIReducer(gpu_context,
                                                                    compressor,
                                                                    communicator) {
  int64_t chunk_size = tensor_fusion_size_;
  chunk_size = utils::aligned_size((chunk_size + world_size - 1) / world_size);
  int64_t buffer_size = chunk_size * world_size +
      chunk_size * (world_size - 1);

  buffer_ = std::make_unique<PersistentBuffer>(buffer_size);
  void *buffer_data = buffer_->RawPointer();
  gradients_send_ = static_cast<unsigned char *>(buffer_data);
  gradients_recv_ = gradients_send_ + chunk_size * world_size;
  gpu_context->StreamCreate(&stream_);
}

int MPI_Allreduce_Ring::AllreduceDivision(int num_elements, int global_offset,
                                          std::vector<Layer> &tensors,
                                          void *comm_p, bool do_compression) {
  int status;
  if (do_compression) {
    status = AllreduceDivisionCompressed(num_elements,
                                         global_offset,
                                         tensors,
                                         comm_p);
  } else {
    status = AllreduceDivisionUncompressed(num_elements,
                                           global_offset,
                                           tensors,
                                           comm_p);
  }
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  MPI_CHECK(MPI_Barrier(comm));
  return status;
}

int MPI_Allreduce_Ring::AllreduceDivisionUncompressed(int num_elements,
                                                      int global_offset,
                                                      std::vector<Layer> &layers,
                                                      void *comm_p) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  std::vector<int> chunk_sizes, offsets;
  unsigned char *send_buf = gradients_send_;
  unsigned char *send_buf_base = send_buf;
  unsigned char *recv_buf = gradients_recv_;
  gpuStream_t stream = stream_;
  int element_size = layers[0].element_size();

  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  Quantizer::GetSizesAndOffsets(num_elements, world_size, global_offset, layers, offsets,
                                chunk_sizes);
  communicator_->Init(world_size, comm_p);
  if (layers.size() > 1) {
    for (auto &layer: layers) {
      gpu_context_->MemcpyAsyncD2D(send_buf,
                                   layer.data_ptr(),
                                   layer.numel() * element_size,
                                   stream);
      send_buf += layer.numel() * element_size;
    }
    send_buf = send_buf_base;
  } else {
    send_buf = static_cast<unsigned char *>(layers[0].data_ptr())
        + element_size * global_offset;
    send_buf_base = send_buf;
  }
  gpu_context_->StreamSynchronize(stream);

  // Receive from your left neighbor with wrap-around.
  const size_t recv_from = (rank - 1 + world_size) % world_size;
  // Send to your right neighbor with wrap-around.
  const size_t send_to = (rank + 1) % world_size;

  int recv_segment_idx, send_segment_idx;
  int buf_send_idx, buf_recv_idx;
  int send_size, recv_size;
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i - 1 + world_size) % world_size;
    send_segment_idx = (rank - i + world_size) % world_size;
    buf_send_idx = offsets[send_segment_idx];
    buf_recv_idx = offsets[recv_segment_idx];
    communicator_->ISend(send_buf + buf_send_idx * element_size,
                         chunk_sizes[send_segment_idx] * element_size,
                         send_to,
                         stream);
    communicator_->IRecv(recv_buf,
                         chunk_sizes[recv_segment_idx] * element_size,
                         recv_from,
                         stream);
    communicator_->WaitSend(send_to);
    communicator_->WaitRecv(recv_from);
    Compressor::Add(chunk_sizes[recv_segment_idx],
                    send_buf + buf_recv_idx * element_size,
                    recv_buf,
                    send_buf + buf_recv_idx * element_size,
                    layers[0].scalar_type(),
                    stream);
  }
  unsigned char *compressed_buf = recv_buf;
  for (int i = 0; i < world_size - 1; i++) {
    send_segment_idx = (rank + world_size + 1) % world_size;
    buf_send_idx = offsets[send_segment_idx];
    send_buf = send_buf_base + buf_send_idx * element_size;
    send_size = chunk_sizes[send_segment_idx] * element_size;

    recv_segment_idx = (rank - i + world_size) % world_size;
    buf_recv_idx = offsets[recv_segment_idx];
    recv_size = chunk_sizes[recv_segment_idx] * element_size;
    recv_buf = send_buf_base + buf_recv_idx * element_size;

    communicator_->ISend(send_buf, send_size, send_to, stream);
    communicator_->IRecv(recv_buf, recv_size, recv_from, stream);
    send_buf += send_size;
    communicator_->WaitSend(send_to);
    communicator_->WaitRecv(recv_from);
  }
  send_buf = send_buf_base;
  if (layers.size() > 1) {
    for (auto &layer: layers) {
      gpu_context_->MemcpyAsyncD2D(layer.data_ptr(),
                                   send_buf,
                                   layer.numel() * element_size,
                                   stream);
      send_buf += layer.numel() * element_size;
    }
  }
  gpu_context_->StreamSynchronize(stream);
  return 0;
}

int MPI_Allreduce_Ring::AllreduceDivisionCompressed(int num_elements,
                                                      int global_offset,
                                                      std::vector<Layer> &layers,
                                                      void *comm_p) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  std::vector<int> chunk_sizes, offsets;
  gpuStream_t stream = stream_;
  Quantizer::GetSizesAndOffsets(num_elements, world_size, global_offset, layers, offsets,
                                chunk_sizes);
  communicator_->Init(world_size, comm_p);
  compressor_->Init(layers[0].element_size(), stream);
  int start_elem = offsets[rank] + global_offset;
  int recv_num_elems = chunk_sizes[rank];
  int recv_compressed_size = utils::aligned_size(
      compressor_->BufferSize(recv_num_elems, layers, start_elem));
  int send_num_elems = 0;
  int send_compressed_size = 0;
  unsigned char *send_buf = gradients_send_;
  unsigned char *recv_buf = gradients_recv_;
  int element_size = layers[0].element_size();
  // Receive from your left neighbor with wrap-around.
  const size_t recv_from = (rank - 1 + world_size) % world_size;
  // Send to your right neighbor with wrap-around.
  const size_t send_to = (rank + 1) % world_size;
  int recv_segment_idx, send_segment_idx;
  int buf_send_idx, buf_recv_idx;
  int send_size, recv_size;

  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i - 1 + world_size) % world_size;
    send_segment_idx = (rank - i + world_size) % world_size;
    buf_send_idx = offsets[send_segment_idx];
    buf_recv_idx = offsets[recv_segment_idx];

    recv_size = utils::aligned_size(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], layers, buf_recv_idx));
    communicator_->IRecv(gradients_recv_, recv_size, recv_from, stream);

    send_size = utils::aligned_size(compressor_->Compress(
        gradients_send_, layers, buf_send_idx, chunk_sizes[send_segment_idx],
        stream));
    communicator_->ISend(gradients_send_, send_size, send_to, stream);
    communicator_->WaitRecv(recv_from);
    communicator_->WaitSend(send_to);
    compressor_->Decompress(gradients_recv_, layers, buf_recv_idx,
                            chunk_sizes[recv_segment_idx], true, stream);
  }

  send_segment_idx = (rank + world_size + 1) % world_size;
  buf_send_idx = offsets[send_segment_idx];
  send_buf = gradients_send_;
  send_size = utils::aligned_size(compressor_->Compress(
      send_buf, layers, buf_send_idx, chunk_sizes[send_segment_idx], stream));
  compressor_->Decompress(send_buf, layers, buf_send_idx,
                          chunk_sizes[send_segment_idx], false, stream);
  recv_buf = send_buf + send_size;
  unsigned char* compressed_buf = recv_buf;
  gpu_context_->StreamSynchronize(stream);

  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    buf_recv_idx = offsets[recv_segment_idx];
    recv_size = utils::aligned_size(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], layers, buf_recv_idx));
    communicator_->ISend(send_buf, send_size, send_to, stream);
    communicator_->IRecv(recv_buf, recv_size, recv_from, stream);
    communicator_->WaitSend(send_to);
    communicator_->WaitRecv(recv_from);
    send_buf += send_size;
    recv_buf += recv_size;
    send_size = recv_size;
  }

  // Decompress all chunks we received.
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    buf_recv_idx = offsets[recv_segment_idx];

    compressor_->Decompress(compressed_buf, layers, buf_recv_idx,
                            chunk_sizes[recv_segment_idx], false, stream);
    recv_size = utils::aligned_size(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], layers, buf_recv_idx));
    compressed_buf += recv_size;
  }
  gpu_context_->StreamSynchronize(stream);
  return 0;
}

} // namespace common
} // namespace cgx
