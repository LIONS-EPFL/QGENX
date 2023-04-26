kubectl delete deployments.apps "qgegdeploy$NUM_PODS"
zsh build_image_local.sh
echo "NUM_PODS=$NUM_PODS"
kubectl apply -f gpu$NUM_PODS.yaml
export FOO=`kubectl get po | rg Running | wc -l`
while [ "$FOO" -lt "$NUM_PODS" ] 
do 
export FOO=`kubectl get po | rg Running | wc -l`
  echo "Wating for pods to launch have $FOO"
  sleep 2
done
zsh copy_experiment_files.sh
zsh connect_to_head_pod.sh
