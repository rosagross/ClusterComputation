#!/usr/bin/env julia

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Statistics
using DataFrames
using CairoMakie
using UnfoldSim
using UnfoldMixedModels
using MixedModelsPermutations
using MAT
using Logging
import Logging: NullLogger

include(joinpath(@__DIR__, "..", "read_adjacency.jl"))
include(joinpath(@__DIR__, "..", "cluster_mass_test.jl"))

mkpath(joinpath(@__DIR__, "figures"))

println("Step 1/7: Setup channel layout and adjacency...")
hart = Hartmut()
channels_1020 = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8", "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
]

all_labels = hart.electrodes["label"]
channel_idx_raw = [findfirst(==(ch), all_labels) for ch in channels_1020]
found_mask = .!isnothing.(channel_idx_raw)
channel_idx = Int[idx for idx in channel_idx_raw if !isnothing(idx)]
selected_labels = channels_1020[found_mask]

neighbor_file = joinpath(@__DIR__, "biosemi32_neighb.mat")
adjacency, _ = read_fieldtrip_neighbors(neighbor_file; channel_order = selected_labels)

fig_adj = Figure(size = (900, 400))
ax_adj_m = Axis(fig_adj[1, 1], title = "Step 1: Channel adjacency matrix", xlabel = "Channel", ylabel = "Channel")
hm_adj = heatmap!(ax_adj_m, adjacency)
Colorbar(fig_adj[1, 2], hm_adj, label = "Adjacent")

pos3d = hart.electrodes["pos"][channel_idx, :]
ax_adj_g = Axis(fig_adj[1, 3], title = "Step 1: 2D sensor graph", xlabel = "x", ylabel = "y", aspect = DataAspect())
for i in 1:size(adjacency, 1)
    for j in (i + 1):size(adjacency, 2)
        if adjacency[i, j]
            lines!(ax_adj_g, [pos3d[i, 1], pos3d[j, 1]], [pos3d[i, 2], pos3d[j, 2]], color = (:gray, 0.5), linewidth = 1)
        end
    end
end
scatter!(ax_adj_g, pos3d[:, 1], pos3d[:, 2], color = :steelblue, markersize = 10)

save(joinpath(@__DIR__, "figures", "step1_adjacency.png"), fig_adj)

println("Step 2/7: Simulate multichannel ERP data with UnfoldSim...")
sfreq = 50
n_subjects = 15
n_items = 30

design = MultiSubjectDesign(
    ;
    n_subjects = n_subjects,
    n_items = n_items,
    items_between = Dict(:stimtype => ["car", "face"]),
)
contrasts = Dict(:stimtype => DummyCoding())

p1 = MixedModelComponent(
    ;
    basis = UnfoldSim.p100(; sfreq = sfreq),
    formula = @formula(dv ~ 1 + (1 | subject) + (1 | item)),
    β = [5.0],
    σs = Dict(:subject => [1.0], :item => [0.5]),
    contrasts = contrasts,
)
n1 = MixedModelComponent(
    ;
    basis = UnfoldSim.n170(; sfreq = sfreq),
    formula = @formula(dv ~ 1 + stimtype + (1 + stimtype | subject) + (1 | item)),
    β = [1.0, 3.0],
    σs = Dict(:subject => [1.5, 0.3], :item => [0.5]),
    contrasts = contrasts,
)
p3 = MixedModelComponent(
    ;
    basis = UnfoldSim.p300(; sfreq = sfreq),
    formula = @formula(dv ~ 1 + (1 | subject) + (1 | item)),
    β = [4.0],
    σs = Dict(:subject => [1.0], :item => [0.5]),
    contrasts = contrasts,
)

mc_p1 = MultichannelComponent(p1, hart => "Right Occipital Pole")
mc_n1 = MultichannelComponent(n1, hart => "Left Occipital Fusiform Gyrus")
mc_p3 = MultichannelComponent(p3, hart => "Right Cingulate Gyrus, posterior division")

data4d, evts = UnfoldSim.simulate(
    MersenneTwister(42),
    design,
    [mc_p1, mc_n1, mc_p3],
    UniformOnset(sfreq * 2, 10),
    PinkNoise(; noiselevel = 1.0);
    return_epoched = true,
)

nch, nt, ntrials_per_sub, nsub = size(data4d)
data3d = reshape(data4d, nch, nt, ntrials_per_sub * nsub)
data = data3d[channel_idx, :, :]
times = range(-0.1, 0.5, length = size(data, 2))

fig_data = Figure(size = (900, 400))
ax_data1 = Axis(fig_data[1, 1], title = "Step 2: mean over trials", xlabel = "Time index", ylabel = "Channel")
hm_data = heatmap!(ax_data1, dropdims(mean(data; dims = 3), dims = 3))
Colorbar(fig_data[1, 2], hm_data, label = "Amplitude")

ax_data2 = Axis(fig_data[1, 3], title = "Step 2: trial sample at channel T7", xlabel = "Time (s)", ylabel = "Amplitude")
t7_idx = findfirst(==("T7"), selected_labels)
if !isnothing(t7_idx)
    lines!(ax_data2, times, data[t7_idx, :, 1], color = :darkorange)
end

save(joinpath(@__DIR__, "figures", "step2_simulated_data.png"), fig_data)

println("Step 3/7: Plot condition ERPs...")
face_trials = findall(evts.stimtype .== "face")
car_trials = findall(evts.stimtype .== "car")

fig_erp = Figure(size = (700, 450))
ax_erp = Axis(fig_erp[1, 1], title = "Step 3: ERP at T7", xlabel = "Time (s)", ylabel = "Amplitude")
if !isnothing(t7_idx)
    erp_face = vec(mean(data[t7_idx, :, face_trials]; dims = 2))
    erp_car = vec(mean(data[t7_idx, :, car_trials]; dims = 2))
    lines!(ax_erp, times, erp_face, color = :firebrick, label = "face")
    lines!(ax_erp, times, erp_car, color = :royalblue, label = "car")
    axislegend(ax_erp, position = :rb)
end
save(joinpath(@__DIR__, "figures", "step3_erp_face_vs_car.png"), fig_erp)

println("Step 4/7: Fit mass-univariate LMM...")
m = fit(
    UnfoldModel,
    [Any => (@formula(0 ~ 1 + stimtype + (1 | item) + (1 | subject)), times)],
    evts,
    data,
)

coefs = coeftable(m)
coef_stim = filter(r -> r.coefname == "stimtype: face", coefs)

estimate_vec = [isnothing(x) ? NaN : Float64(x) for x in coef_stim.estimate]
stderror_vec = [isnothing(x) ? NaN : Float64(x) for x in coef_stim.stderror]
z_vec = estimate_vec ./ stderror_vec
coef_stim[!, :z] = z_vec

z_summary = combine(groupby(coef_stim, [:channel, :time]), :z => (x -> mean(filter(!isnan, x))) => :z)
sort!(z_summary, [:channel, :time])
z_summary.z = replace(z_summary.z, NaN => 0.0)

unique_channels = sort(unique(z_summary.channel))
unique_times = sort(unique(z_summary.time))
z_map = reshape(z_summary.z, length(unique_times), length(unique_channels))'

fig_z = Figure(size = (900, 400))
ax_z = Axis(fig_z[1, 1], title = "Step 4: Uncorrected z-map (stimtype)", xlabel = "Time (s)", ylabel = "Channel")
hm_z = heatmap!(ax_z, unique_times, 1:length(unique_channels), z_map', colormap = :RdBu, colorrange = (-4, 4))
Colorbar(fig_z[1, 2], hm_z, label = "z")
contour!(ax_z, unique_times, 1:length(unique_channels), abs.(z_map)', levels = [2.0], color = :black, linewidth = 1)
save(joinpath(@__DIR__, "figures", "step4_uncorrected_zmap.png"), fig_z)

println("Step 5/7: Compute permutation statistics...")
coefficient = 2
n_permutations = 50
threshold = 2.0
permuted = lmm_permutations(
    MersenneTwister(42),
    m,
    data,
    coefficient;
    n_permutations = n_permutations,
    lmm_statistic = :z,
    time_selection = 1:length(times),
)

null_max = zeros(n_permutations)
for p in 1:n_permutations
    _, perm_stats = find_connected_clusters_2d(permuted[:, :, p], adjacency, threshold)
    null_max[p] = isempty(perm_stats) ? 0.0 : maximum(perm_stats)
end

fig_perm = Figure(size = (700, 450))
ax_perm = Axis(fig_perm[1, 1], title = "Step 5: Null max-cluster distribution", xlabel = "Max cluster mass", ylabel = "Count")
hist!(ax_perm, null_max, bins = 20, color = (:gray, 0.7), strokecolor = :black)
save(joinpath(@__DIR__, "figures", "step5_null_distribution.png"), fig_perm)

println("Step 6/7: Compute cluster-corrected p-values...")
pvals = spatiotemporal_cluster_pvalues(MersenneTwister(1), z_map, permuted, adjacency, threshold)

fig_p = Figure(size = (900, 400))
ax_p = Axis(fig_p[1, 1], title = "Step 6: Cluster-corrected p-values", xlabel = "Time (s)", ylabel = "Channel")
hm_p = heatmap!(ax_p, unique_times, 1:length(unique_channels), pvals', colormap = :viridis, colorrange = (0, 0.2))
Colorbar(fig_p[1, 2], hm_p, label = "p")
contour!(ax_p, unique_times, 1:length(unique_channels), pvals', levels = [0.05], color = :red, linewidth = 2)
save(joinpath(@__DIR__, "figures", "step6_cluster_corrected_pvalues.png"), fig_p)

println("Step 7/7: Visualize significant cluster mask...")
sig_mask = pvals .< 0.05
fig_sig = Figure(size = (900, 400))
ax_sig = Axis(fig_sig[1, 1], title = "Step 7: Significant mask (p < 0.05)", xlabel = "Time (s)", ylabel = "Channel")
hm_sig = heatmap!(ax_sig, unique_times, 1:length(unique_channels), Float64.(sig_mask)', colormap = [:white, :seagreen])
Colorbar(fig_sig[1, 2], hm_sig, label = "Significant")
save(joinpath(@__DIR__, "figures", "step7_significant_mask.png"), fig_sig)

println("Done. Figures saved in: $(joinpath(@__DIR__, "figures"))")
