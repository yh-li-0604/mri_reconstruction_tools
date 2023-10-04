using PythonCall
using EllipsisNotation
using Random
using Strided, Base.Threads
using Test, BenchmarkTools

function change_array_layout(arr::AbstractArray)
	# change array column major to row major
	# for python program to call
	permutedims(arr, reverse(1:ndims(arr)))
end

""" 
data is a array with size (l, spokes_num, ...)
output is a array with size (l, spokes_per_phase, phase_num, ...) 
"""
function data_binning(data::AbstractArray{T}, sorted_r_idx::AbstractVector{Int}, phase_num::Int, spokes_per_phase::Int)::AbstractArray{T} where {T}
    # l, sp, t, n = spoke_len, spokes_per_contra, contrast_num, n_samples
    s = size(data)
    data_sorted = selectdim(data, 2, sorted_r_idx)
    output = reshape(data_sorted, s[1], spokes_per_phase, phase_num, s[3:end]...) |> # reshape to original batch shape
             (x -> permutedims(x, (1, 2, 4:ndims(x)..., 3)))
    return output
end

""" 
data is a array with size (l, spokes_num, ...)
output is a array with size (l, spokes_per_phase, phase_num, contrast_num, ...)
"""
function data_binning(data::AbstractArray{T}, sorted_r_idx::AbstractArray, contrast_num::Int, spokes_per_contra::Int, phase_num::Int, spokes_per_phase::Int)::AbstractArray{T} where {T}
    s = size(data)
    data_t = reshape(data, s[1], spokes_per_phase * phase_num, contrast_num, s[3:end]...)
    data_ph_t = Array{T}(undef, s[1], spokes_per_phase, s[3:end]..., phase_num, contrast_num)
    @threads for i in 1:contrast_num
        data_ph_t[..,i] = data_binning(selectdim(data_t,3,i), selectdim(sorted_r_idx,2,i), phase_num, spokes_per_phase)
    end
    return data_ph_t
end

function data_binning(data::PyArray, sorted_r_idx::PyArray, contrast_num::Int, spokes_per_contra::Int, phase_num::Int, spokes_per_phase::Int)
    data = data |> change_array_layout
    sorted_r_idx = sorted_r_idx |> change_array_layout 
    data_binning(data, sorted_r_idx.+1, contrast_num, spokes_per_contra, phase_num, spokes_per_phase) |> change_array_layout 
end

# data = rand(640, 2550, 80, 15)
# contrast_num = 34
# sorted_r_idx = randperm(75) |> x -> repeat(x, 1, contrast_num)
# spokes_per_contra = 75
# phase_num = 5
# spokes_per_phase = 15
# @time jl_output = data_binning(data, sorted_r_idx, contrast_num, spokes_per_contra, phase_num, spokes_per_phase);


