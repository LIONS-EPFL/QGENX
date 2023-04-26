# get the pod ips
kubectl get pod  -o wide | awk  'NR>1 {print $6 " slots=1"}' > hostfile
cat hostfile
MASTER_HOST=$(cat hostfile | awk 'NR==1 {print $1}')
echo "export MASTER_HOST=$MASTER_HOST" > master_def
tar vczf train_files.tar.gz train_extraadam.py dist_mpi_launch.sh config models utils.py optim master_def wandbkey nuq normallaunch.sh mpilaunch.sh hostfile
for pod in $(kubectl get pod |awk 'NR>1 {print $1}')
do
    echo "copying to $pod"
    kubectl exec -ti $pod -- bash -c 'rm -rf "/root/.ssh/" && mkdir -p "/root/.ssh/"'
    for f in qgqg_ed25519.pub qgqg_ed25519 authorized_keys
    do
       echo "Copying $f"
       kubectl cp $f $pod:/root/.ssh/$f
    done
    kubectl cp ./user_ssh_conf $pod:/root/.ssh/config
    kubectl exec -ti $pod -- chown -R root /root/.ssh
    kubectl exec -ti $pod -- chmod  600 /root/.ssh/qgqg_ed25519 /root/.ssh/qgqg_ed25519.pub /root/.ssh/config
    kubectl exec -ti $pod -- bash -c "echo 'eval \`ssh-agent -s\`' >> /root/.bashrc"
    kubectl exec -ti $pod -- bash -c "echo 'ssh-add /root/.ssh/qgqg_ed25519' >> /root/.bashrc"
    f=train_files.tar.gz
    #for f in train_files.tar.gz do
     echo "Copying $f"
     kubectl cp $f $pod:/root/$f
     kubectl exec -ti $pod -- chown root /root/$f
     kubectl exec -ti $pod -- bash -c "cd /root && tar xf $f && chown -R root /root/"
    #done
    #kubectl exec -ti $pod -- bash -c 'echo "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi" >> /root/.bash_profile'
    #kubectl exec -ti $pod -- bash -c 'echo "source /opt/cond/bin/activate" >> /root/.bashrc'
done
