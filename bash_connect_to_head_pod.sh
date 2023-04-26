kubectl exec -ti $(kubectl get pod | tail -n 1 | awk '{ print $1}') -- bash
