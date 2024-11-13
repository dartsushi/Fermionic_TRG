n_iter = 8
system_size = 2^(n_iter - 1)
input_mu = 0
input_mass = 0

series_1 = collect(1:system_size)
series_2 = collect(1:system_size)
p_1 = zeros(ComplexF64, system_size)
p_2 = zeros(ComplexF64, system_size)

a_11 = zeros(ComplexF64, (system_size, system_size))
a_12 = zeros(ComplexF64, (system_size, system_size))
a_21 = zeros(ComplexF64, (system_size, system_size))
a_22 = zeros(ComplexF64, (system_size, system_size))
det_D = zeros(ComplexF64, (system_size, system_size))

exact = 0

for i in 1:system_size
    p_1[i] = 2π * series_1[i] / system_size

    p_2[i] = 2π * series_2[i] / system_size
    p_2[i] = p_2[i] + π / system_size
end

for j in 1:system_size
    for i in 1:system_size
        a_11[i, j] = -cos(p_1[i]) - cos(p_2[j]) * cosh(input_mu) - im * sin(p_2[j]) * sinh(input_mu) + input_mass +2   
        a_22[i, j] = a_11[i, j]
        a_12[i, j] = 1im * sin(p_1[i]) + sin(p_2[j]) * cosh(input_mu) - 1im * cos(p_2[j]) * sinh(input_mu)
        a_21[i, j] = 1im * sin(p_1[i]) - sin(p_2[j]) * cosh(input_mu) + 1im * cos(p_2[j]) * sinh(input_mu)
        det_D[i, j] = a_11[i, j] * a_22[i, j] - a_12[i, j] * a_21[i, j]
    end
end


for j in 1:system_size
    for i in 1:system_size
        global exact = exact + log(det_D[i,j])/system_size^2
    end
end

println(exact)