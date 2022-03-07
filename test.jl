using Flux

h = rand(Float32,2,3)

m = rand(Float32,2,3)

DNN = Chain(
    x->transpose(x),
    Dense(3,12,relu),
    Dense(12,8,relu),
    Dense(8,3,sigmoid),
    x->transpose(x)
)
Flux.train!((x,y)->Flux.Losses.mse(DNN(x),y),Flux.params(DNN),[(h,m)],Flux.Optimise.ADAM())