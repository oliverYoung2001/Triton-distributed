# Print cluster states
sinfo -N --Format=NodeHost,StateLong,CPUsState,Gres,GresUsed
sinfo -N --Format=NodeHost:6,StateLong:6,CPUsState:16,Gres:40,GresUsed:40
sinfo -N -o "%.24N %.18T %.20C %.80G %.80g"