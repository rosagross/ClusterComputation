#!/usr/bin/env julia
# Converted from tutorial_notebook.ipynb — linear script for debugging


import Pkg
Pkg.activate("/data/p_02865/tools/UnfoldStats.jl/docs")


using UnfoldSim
using MAT
using CairoMakie
using Random
using Statistics
using UnfoldMixedModels
#using UnfoldStats
using DataFrames
using MixedModelsPermutations
using UnfoldMakie

include("/data/hu_steinfath/Desktop/Code/ClusterComputation/read_adjacency.jl")

############################################################
# Load head model and define electrode layout
############################################################
println("Loading head model and defining channels...")
hart = Hartmut()

# Define 10-20 channels (these fit to the biosemi neighbours file we later use)
standard_1020 = ["Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8", "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz"]

# Load FieldTrip neighbor file and reorder to match our channel layout
ft_neighbor_path = "/data/u_steinfath_software/Matlab_tooloxes/fieldtrip-20230118/template/neighbours/biosemi32_neighb.mat"

adjacency, matched_labels = read_fieldtrip_neighbors(
    ft_neighbor_path; 
    channel_order = standard_1020
)

############################################################
# Visualize the adjacency (optional; displays a 3D figure)
############################################################
println("Preparing adjacency visualization...")
all_labels = hart.electrodes["label"]
channel_idx_raw = [findfirst(==(ch), all_labels) for ch in standard_1020]
found_mask = .!isnothing.(channel_idx_raw)
channel_idx = Int64[idx for idx in channel_idx_raw if !isnothing(idx)]
electrode_labels = standard_1020[found_mask]

# Extract subset of electrode positions
electrode_positions = hart.electrodes["pos"][channel_idx, :]

fig = Figure(size = (900, 700))
ax = Axis3(fig[1, 1], 
    title = "10-20 Electrode Adjacency",
    xlabel = "X", ylabel = "Y", zlabel = "Z",
    aspect = :data)

pos3d = electrode_positions

for i in 1:size(adjacency, 1)
    for j in (i+1):size(adjacency, 2)
        if adjacency[i, j]
            lines!(ax, 
                [pos3d[i, 1], pos3d[j, 1]], 
                [pos3d[i, 2], pos3d[j, 2]], 
                [pos3d[i, 3], pos3d[j, 3]],
                color = (:gray, 0.5), linewidth = 1.0)
        end
    end
end

scatter!(ax, pos3d[:, 1], pos3d[:, 2], pos3d[:, 3], 
    markersize = 12, color = :steelblue)

for (i, label) in enumerate(electrode_labels)
    text!(ax, pos3d[i, 1], pos3d[i, 2], pos3d[i, 3] + 0.01, 
        text = label, fontsize = 8, align = (:center, :bottom))
end

display(fig)

############################################################
# 2. Simulate EEG data
############################################################
println("Setting up simulation parameters...")
sfreq = 50  # sampling frequency (Hz)
n_subjects = 15
n_items = 30 # per condition

# Experimental design: face vs car
exp_design = MultiSubjectDesign(; 
    n_subjects = n_subjects,
    n_items = n_items,
    items_between = Dict(:stimtype => ["car", "face"]),
)

contrasts = Dict(:stimtype => DummyCoding())

print(exp_design)

############################################################
# Define ERP components
############################################################
# P100 - no condition effect
p1 = MixedModelComponent(; 
    basis = UnfoldSim.p100(; sfreq = sfreq), # positive peak at 100ms
    formula = @formula(dv ~ 1 + (1 | subject) + (1 | item)),
    β = [5.0],
    σs = Dict(:subject => [1.0], :item => [0.5]),
    contrasts = contrasts,
);

# N170 - WITH condition effect (faces more negative than cars)
n1 = MixedModelComponent(; 
    basis = UnfoldSim.n170(; sfreq = sfreq),
    formula = @formula(dv ~ 1 + stimtype + (1 + stimtype | subject) + (1 | item)),
    β = [1.0, 3],
    σs = Dict(:subject => [1.5, 0.3], :item => [0.5]),
    contrasts = contrasts,
);

# P300 - no condition effect
p3 = MixedModelComponent(; 
    basis = UnfoldSim.p300(; sfreq = sfreq),
    formula = @formula(dv ~ 1 + (1 | subject) + (1 | item)),
    β = [4.0],
    σs = Dict(:subject => [1.0], :item => [0.5]),
    contrasts = contrasts,
);

############################################################
# Project components to scalp & simulate
############################################################
mc_p1 = MultichannelComponent(p1, hart => "Right Occipital Pole")
mc_n1 = MultichannelComponent(n1, hart => "Left Occipital Fusiform Gyrus")
mc_p3 = MultichannelComponent(p3, hart => "Right Cingulate Gyrus, posterior division")

println("Simulating data (this may take some time)...")
@time data_full, evts = UnfoldSim.simulate(
    MersenneTwister(42),
    exp_design,
    [mc_p1, mc_n1, mc_p3],
    UniformOnset(sfreq * 2, 10),
    PinkNoise(; noiselevel = 1.0);
    return_epoched = true,
)

# Handle 4D array: (channels, times, items, subjects) -> reshape to (channels, times, trials)
n_ch_full, n_times_full, n_items_full, n_subjects_full = size(data_full)
data_full_3d = reshape(data_full, n_ch_full, n_times_full, n_items_full * n_subjects_full)

# Extract only the 10-20 channels we selected earlier
data_e = data_full_3d[channel_idx, :, :]

n_channels = size(data_e, 1)
n_times = size(data_e, 2)
n_trials = size(data_e, 3)
times = range(-0.1, 0.5, length = n_times)

############################################################
# Quick ERP plot at channel T7
############################################################
println("Plotting example ERP for T7 (face vs car)...")
pz_idx = findfirst(==("T7"), electrode_labels)
face_trials = findall(evts.stimtype .== "face")
car_trials = findall(evts.stimtype .== "car")
erp_face = mean(data_e[pz_idx, :, face_trials]; dims=2)[:]
erp_car = mean(data_e[pz_idx, :, car_trials]; dims=2)[:]

fig = Figure(size=(600,400))
ax = Axis(fig[1,1], xlabel="Time (s)", ylabel="Amplitude (a.u.)", title="ERP at T7: Face vs Car")
lines!(ax, times, erp_face, color=:red, label="Face")
lines!(ax, times, erp_car, color=:blue, label="Car")
axislegend(ax)
display(fig)

############################################################
# Fit mass-univariate LMM
############################################################
println("Fitting LMM at each channel × time point...")
@time m = fit(
    UnfoldModel,
    [
        Any => (
            @formula(0 ~ 1 + stimtype + (1 | item) + (1 | subject)),
            times,
        ),
    ],
    evts,
    data_e,
);

############################################################
# 3. Perform cluster-mass permutation test
############################################################
import Logging: NullLogger
include("../cluster_mass_test.jl")  # adjust path if needed

coefficient = 2  # stimtype effect (face vs car)
n_permutations = 50  # Reduced for dev; use 500-1000 for real analyses
threshold = 2.0

time_selection = 1:length(times)

println("Generating permutation distribution...")
permuted = lmm_permutations(MersenneTwister(42), m, data_e, coefficient; 
    n_permutations = n_permutations,
    lmm_statistic = :z,
    time_selection = time_selection
)

############################################################
# Check uncorrected effects before running permutation test
############################################################
println("Extracting coefficient table and computing z-values...")
coefs = coeftable(m)
println("Available columns: $(names(coefs))")
coef_stimtype = filter(r -> r.coefname == "stimtype: face", coefs)
println("Rows for stimtype: $(nrow(coef_stimtype))")

unique_channels = sort(unique(coef_stimtype.channel))
unique_times = sort(unique(coef_stimtype.time))
println("Unique channels: $(length(unique_channels))")
println("Unique times: $(length(unique_times))")

# Compute z-values, handling missing values
estimate_vec = [isnothing(x) ? NaN : Float64(x) for x in coef_stimtype.estimate]
stderror_vec = [isnothing(x) ? NaN : Float64(x) for x in coef_stimtype.stderror]
z_vec = estimate_vec ./ stderror_vec
coef_stimtype[!, :z] = z_vec

z_summary = combine(
    groupby(coef_stimtype, [:channel, :time]),
    :z => (x -> mean(filter(!isnan, x))) => :z,
)
sort!(z_summary, [:channel, :time])
z_summary.z = replace(z_summary.z, NaN => 0.0)

n_ch_coef = length(unique_channels)
n_t_coef = length(unique_times)
# Note: reshape expects column-major ordering consistent with how z_summary was constructed
z_vals = reshape(z_summary.z, n_t_coef, n_ch_coef)'  # (channels × times)

n_sig_uncorrected = sum(abs.(z_vals) .> 2.0)
max_z = maximum(abs.(z_vals))
peak_idx = argmax(abs.(z_vals))
peak_ch, peak_t = Tuple(CartesianIndices(z_vals)[peak_idx])

println("\n=== Uncorrected Effects (before permutation) ===")
println("Max |z|: $(round(max_z, digits=2))")
println("Peak location: channel $(electrode_labels[peak_ch]) at t = $(round(unique_times[peak_t], digits=3))s")
println("Samples with |z| > 2.0: $(n_sig_uncorrected) / $(length(z_vals))")
println("Samples with |z| > 1.96: $(sum(abs.(z_vals) .> 1.96)) (uncorrected p < 0.05)")

############################################################
# Visualize uncorrected z-values
############################################################
fig_z = Figure(size = (800, 400))
ax_z = Axis(fig_z[1, 1], 
    title = "Uncorrected z-values for stimtype effect",
    xlabel = "Time (s)",
    ylabel = "Channel")
hm = heatmap!(ax_z, collect(unique_times), 1:n_ch_coef, z_vals', colormap = :RdBu, colorrange = (-4, 4))
Colorbar(fig_z[1, 2], hm, label = "z-value")
display(fig_z)

############################################################
# 3. Run cluster-mass permutation test and visualize corrected p-values
############################################################
println("Computing cluster-mass corrected p-values...")
observed = z_vals;
pvals_clustermass = spatiotemporal_cluster_pvalues(MersenneTwister(1), observed, permuted, adjacency, threshold)

fig = Figure(size = (1400, 500))
ax1 = Axis(fig[1, 1], title = "Observed |z|-values", xlabel = "Time (s)", ylabel = "Channel")
hm1 = heatmap!(ax1, collect(times), 1:n_channels, abs.(z_vals)', colormap = :viridis, colorrange = (0, 6))
Colorbar(fig[1, 2], hm1, label = "|z|")
contour!(ax1, collect(times), 1:n_channels, abs.(z_vals)', levels = [2.0], color = :red, linewidth = 2)

ax2 = Axis(fig[1,3], title="Cluster-mass corrected p-values", xlabel="Time (s)", ylabel="Channel", yticks=(1:n_ch_coef, electrode_labels))
hm_p = heatmap!(ax2, collect(times), 1:n_ch_coef, pvals_clustermass', colormap=:viridis, colorrange=(0,0.1))
Colorbar(fig[1,2], hm_p, label="p-value")

display(fig)

println("Significant (p<0.05) mask summary:", sum(pvals_clustermass .< 0.05), " samples")

############################################################
# Topoplot and butterfly plot for selected channels
############################################################
pos3d = hart.electrodes["pos"][channel_idx, :]  # 32x3 matrix for selected channels
pos2d = [Point2f(pos3d[i,1], pos3d[i,2]) for i in 1:size(pos3d,1)]
sig_mask = pvals_clustermass .< 0.05
cluster_data = copy(z_vals)

f = Figure()
# Average ERP across all trials for each channel and time
erp = mean(data_e; dims=3)[:,:,1]  # (channels x times)
df_erp = DataFrame(
    :amplitude => vec(erp),
    :channel => repeat(1:size(erp, 1), outer = size(erp, 2)),
    :time => repeat(1:size(erp, 2), inner = size(erp, 1)),
)
plot_butterfly!(f[1, 1:2], df_erp; positions = pos2d, mapping=(; y=:amplitude))

# Find significant time points (p < 0.05)
sig_time_bool = vec(any(sig_mask, dims=1))
unique_times = sort(unique(df_erp.time))
sig_times = unique_times[sig_time_bool]
if isempty(sig_times)
    @warn "No significant time points found; falling back to averaging across all times"
    sig_times = unique_times
end
println("Averaging across $(length(sig_times)) time points")

# Compute average topography robustly
avg_topo = Float64[]
for ch in 1:n_channels
    sel = (df_erp.channel .== ch) .& in.(df_erp.time, Ref(sig_times))
    vals = df_erp.amplitude[sel]
    if isempty(vals)
        push!(avg_topo, NaN)
    else
        push!(avg_topo, mean(vals))
    end
end
println("Any NaN in avg_topo: ", any(isnan, avg_topo))
if any(isnan, avg_topo)
    @warn "avg_topo contains NaNs for channels: $(findall(isnan, avg_topo))"
end

# Highlight significant channels: a channel is significant if any time point is significant for it
sig_ch = [any(sig_mask[ch, :]) for ch in 1:n_channels]
marker_colors = [is_sig ? :white : :gray for is_sig in sig_ch]
marker_sizes = [is_sig ? 10 : 10 for is_sig in sig_ch]

plot_topoplot!(
    f[2, 1],
    DataFrame(:amplitude => avg_topo, :channel => 1:n_channels);
    positions = pos2d,
    mapping=(; y=:amplitude),
    visual = (; enlarge = 1, label_scatter = true, colormap = :berlin, contours = false),
    colorrange = (; colorrange = (-0.5, 0.5)),
    topo_attributes = (; label_scatter = (; color = marker_colors, markersize = marker_sizes)),
    )

display(f)

println("Script finished: ClusterComputation/notebooks/tutorial_notebook.jl")
