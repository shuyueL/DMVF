for i in {1..5}
do
    for j in {0..5}
    do
        mv split_$i/view_$j split_$i/view_$j.mat
        mv proc_rate_agent_$j_split_$i proc_rate_agent_$j_split_$i.mat
    done
done
