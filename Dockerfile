FROM qgeg_cgx:august

RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server tmux&& \
    mkdir -p /var/run/sshd

ENV MPI_HOME=/opt/hpcx/ompi/
ENV NCCL_INCLUDE=/usr/include
ENV NCCL_LIB=/usr/lib/x86_64-linux-gnu/
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN pip install torch-fidelity wandb tqdm

RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN usermod -p '$1$Oor8qAiG$S7nUy/Q8SdBicf.V3Zo8G0' root

EXPOSE 22
WORKDIR "/root/"
CMD ["/usr/sbin/sshd", "-D"]
# will need to configure the worker nodes with these: https://github.com/kubeflow/mpi-operator/blob/master/examples/v2beta1/pi/pi.yaml
