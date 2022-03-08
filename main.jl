include("Besection.jl")
using Flux
using MAT
using Random
using TensorBoardLogger,Logging

# 结果显示-与python pytorch的tensorboard一样
lg=TBLogger("tensorboard_logs/run", min_level=Logging.Info)

# defind some variables
# 内存大小-存储数据条数
const MEMORY_SIZE = 1024
# 用户设备数量
const DEVICE_NUM = 10
# 优化间隔-DNN的训练间隔
const OPTIM_INTERVAL = 32
# 一次投入数据大小
const BATCH_SIZE = 128
# DNN学习率  ADAM
const LEARNING_RATE = 0.01
const LEARNING_BETAS = (0.09,0.999)

# defind module
# DNN网络
DNN = Chain(
    x->transpose(x),
    Dense(DEVICE_NUM,120,relu),
    Dense(120,80,relu),
    Dense(80,DEVICE_NUM,sigmoid),
    x->transpose(x)
)
# 内存大小 2*DEVICE_NUM表示信道增益与对应的卸载策略的大小
MEMORY = zeros(Float32,MEMORY_SIZE,2*DEVICE_NUM)

# data handle
# 导入数据
INPUT_H = matread("data/data_10.mat")["input_h"].* 10^6
# 损失函数  交叉熵损失
loss(x,y) = Flux.Losses.binarycrossentropy(DNN(x),y)
# 优化器
optimizer = Flux.Optimise.ADAM(LEARNING_RATE,LEARNING_BETAS,0.0001)
# 模型参数
params = Flux.params(DNN)

println("Start training")
# training

with_logger(lg) do
    # 开始训练
    for epoch = 1:size(INPUT_H,1)
        # 获取当前信道增益
        h = INPUT_H[epoch,:]
        # 通过DNN获取卸载决策变量
        xi = DNN(Flux.unsqueeze(h,1))
        # 使用保序法获取决策向量
        mlist = Besection.knm(xi[1,:],10)
        # 获取每一个决策向量对应的计算速率
        r_list = []
        for m = mlist
            push!(r_list,Besection.besection(h/10^6,m)[1])
        end
        # 将最大的速率对应的信道增益与卸载决策存入内存
        MEMORY[(epoch-1)%MEMORY_SIZE + 1,:] =  vcat(h,mlist[argmax(r_list)])
        # optimizer
        # 达到优化间隔进行优化
        if epoch % OPTIM_INTERVAL == 0
            # 判断当前时间帧是否超过内存大小
            if epoch < MEMORY_SIZE
                # 没有，则从当前时间帧之前可重复选择出训练批次大小的下标
                idx = rand(1:epoch,BATCH_SIZE)
            else
                idx = randperm(MEMORY_SIZE)[1:BATCH_SIZE]
            end
            # 从内存中读取数据
            hm = MEMORY[idx,:]
            # 获取信道增益
            h_ = hm[:,1:DEVICE_NUM]
            # 获取卸载决策
            m_ = hm[:,DEVICE_NUM+1:2*DEVICE_NUM]
            # 损失变量
            local loss_v = 0.0
            # 计算梯度
            gs = gradient(params) do
                # 计算损失
                loss_val = loss(h_,m_)
                loss_v = loss_val
                # 返回损失
                return loss_val
            end
            # 更新参数  使用优化器与梯度
            Flux.update!(optimizer,params,gs)
            # 简单绘制结果到tensorboard
            @info "DROO_LOSS" loss=loss_v step = epoch
        end
    end
end


println("close")