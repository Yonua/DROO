module Besection
using LambertW
function phi(v, j, wj, epsilon)
    return 1 / (-1 - 1 / (real(lambertw(-1 / (exp(1 + v / wj[j] / epsilon))))))
end

function p1(v, M1, hj, eta2, wj, epsilon)
    p1 = 0
    for j = 1:length(M1)
        p1 += hj[j]^2 * phi(v, j, wj, epsilon)
    end
    return 1 / (1 + p1 * eta2)
end

function Q(v, wi, wj, eta1, eta2, hi, hj, ki, M1, epsilon)
    sum1 = sum(wi .* eta1 .* (hi / ki) .^ (1.0 / 3)) * p1(v, M1, hj, eta2, wj, epsilon)^(-2 / 3) / 3
    sum2 = Float32(0)
    for j = 1:length(M1)
        sum2 += wj[j] * hj[j]^2/(1 + 1 / phi(v, j, wj, epsilon))
    end
    return sum1 + sum2*epsilon*eta2 - v
end

function tau(v, j, hj,eta2,wj,epsilon,M1)
    return eta2*hj[j]^2*p1(v,M1,hj,eta2,wj,epsilon)*phi(v,j,wj,epsilon)
end

function rate(x,wi,wj,hi,hj,ki,eta1,eta2,M1,epsilon)
    sum1 = sum(wi.*eta1.*(hi/ki).^(1.0/3).*x[1]^(1.0/3))
    sum2 = Float32(0)
    for i = 1:length(M1)
        sum2 += wj[i]*epsilon*x[i+1]*log(1+eta2*hj[i]^2*x[1]/x[i+1])
    end
    return sum1 + sum2
end

function besection(h::Vector, M::Vector, weights::Vector=[])
    # some parameters
    o = 100
    p = 3
    u = 0.7
    eta1 = ((u * p)^(1.0 / 3)) / o
    ki = 10^-26
    eta2 = u * p / 10^-10
    B = 2 * 10^6
    Vu = 1.1
    epsilon = B / (Vu * log(2))
    x = Vector{Float32}()

    # offloading to locally, offloading to edge
    M0, M1 = findall(x -> x == 0, M), findall(x -> x == 1, M)
    hi, hj = h[M0], h[M1]

    N = length(M)
    if length(weights) < N
        weights = ones(Float32, N)
        weights[1:2:N] .= 1.5
    end
    wi = weights[M0]
    wj = weights[M1]

    delta = 0.005
    UB = 999999999
    LB = 0
    v = 0.0
    while UB - LB > delta
        v = (UB + LB) / 2
        if Q(v,wi,wj,eta1,eta2,hi,hj,ki,M1,epsilon) > 0
            LB = v
        else
            UB = v
        end
    end
    push!(x,p1(v,M1,hj,eta2,wj,epsilon))
    for j = 1:length(M1)
        push!(x, tau(v,j,hj,eta2,wj,epsilon,M1))
    end
    return rate(x,wi,wj,hi,hj,ki,eta1,eta2,M1,epsilon),x[1],x[2:end]
end

function knm(m,k=1)
    m_list = []
    push!(m_list,1*(m.>0.5))
    if k > 1
        m_abs = abs.(m .- 0.5)
        idx_list = sortperm(m_abs)[1:k]
        for i = 1:k
            if m[idx_list[i]] > 0.5
                push!(m_list,1*((m .- m[idx_list[i]]) .> 0))
            else
                push!(m_list,1*((m .- m[idx_list[i]]) .>= 0))
            end
        end
    end
    return m_list
end

export besection,knm
end
