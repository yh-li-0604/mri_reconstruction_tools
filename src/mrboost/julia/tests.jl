include("computation.jl")

# begin
    data = rand(640, 2550, 80, 15);
    contrast_num = 34;
    sorted_r_idx = randperm(75) |> x -> repeat(x, 1, contrast_num);
    spokes_per_contra = 75;
    phase_num = 5;
    spokes_per_phase = 15;
    @time jl_output = data_binning(data, sorted_r_idx, contrast_num, spokes_per_contra, phase_num, spokes_per_phase);
    # @test jl_output = data_binning(data, sorted_r_idx, contrast_num, spokes_per_contra, phase_num, spokes_per_phase);
    # comp = pyimport("mrboost.computation")
    # torch = pyimport("torch")
    # np = pyimport("numpy")
    # data = data |> change_array_layout |> np.array |> torch.tensor
    # sorted_r_idx = sorted_r_idx.-1 |> change_array_layout |> np.array |> torch.tensor
    # @time py_output = comp.data_binning_old_version(data, sorted_r_idx, contrast_num, spokes_per_contra, phase_num, spokes_per_phase);
    # @test py_output = comp.data_binning_old_version(data, torch.tensor(sorted_r_idx), contrast_num, spokes_per_contra, phase_num, spokes_per_phase);
    # py_output = pyconvert(Array,py_output) |> change_array_layout
    # @test jl_output ≈ py_output;
# end


