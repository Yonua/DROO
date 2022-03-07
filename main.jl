include("Besection.jl")
using Flux
using MAT
using Random

# besection([1.0,2,3],[1,0,1],[1.0])

# defind variables
const MEMORY_SIZE = 1024
const DEVICE_NUM = 10
const OPTIM_INTERVAL = 32
const BATCH_SIZE = 128
const LEARNING_RATE = 0.01
const LEARNING_BETAS = (0.09,0.999)

# defind module
DNN = Chain(
    x->transpose(x),
    Dense(DEVICE_NUM,120,relu),
    Dense(120,80,relu),
    Dense(80,DEVICE_NUM,sigmoid),
    x->transpose(x)
)

MEMORY = zeros(Float32,MEMORY_SIZE,2*DEVICE_NUM)

# data handle
INPUT_H = matread("data/data_10.mat")["input_h"].* 10^6

optimizer = Flux.Optimise.ADAM(LEARNING_RATE,LEARNING_BETAS,0.0001)
params = Flux.params(DNN)

# training
for epoch = 1:size(INPUT_H,1)
    h = INPUT_H[epoch,:]
    xi = DNN(Flux.unsqueeze(h,1))
    mlist = Besection.knm(xi[1,:],10)
    r_list = []
    for m = mlist
        push!(r_list,Besection.besection(h/10^6,m)[1])
    end
    MEMORY[(epoch-1)%MEMORY_SIZE + 1,:] =  vcat(h,mlist[argmax(r_list)])
    # optimizer
    if epoch % 1 == 0
        if epoch < MEMORY_SIZE
            idx = rand(1:epoch,BATCH_SIZE)
        else
            idx = randperm(MEMORY_SIZE)[1:BATCH_SIZE]
        end
        hm = MEMORY[idx,:]
        h_ = hm[:,1:DEVICE_NUM]
        m_ = hm[:,DEVICE_NUM+1:2*DEVICE_NUM]
        Flux.train!((x,y) -> Flux.Losses.mse(DNN(x),y),params,[(h_,m_)],optimizer)
    end
    if epoch == 2
        break
    end
end