
# """
#     find_connected_clusters_2d(data::AbstractMatrix, adjacency::AbstractMatrix{Bool}, threshold::Real)

# Find spatiotemporal clusters in a 2D data matrix (channels × time) using spatial connectivity.

# A cluster is a set of spatiotemporally connected samples where:
# 1. Each sample exceeds the threshold
# 2. Samples are connected in time (consecutive time points)
# 3. Samples are connected in space (via the adjacency matrix)

# # Arguments
# - `data::AbstractMatrix`: A channels × time matrix of test statistics
# - `adjacency::AbstractMatrix{Bool}`: Spatial adjacency matrix (channels × channels)
# - `threshold::Real`: Cluster-forming threshold

# # Returns
# - `clusters::Vector{Vector{CartesianIndex}}`: Vector of clusters, where each cluster 
#   is a vector of CartesianIndex positions in the data matrix
# - `cluster_stats::Vector{Float64}`: Vector of cluster statistics (sum of values in each cluster)

# # Examples
# ```julia
# data = rand(10, 50)  # 10 channels, 50 time points
# adj, _ = read_fieldtrip_neighbors("biosemi64_neighb.mat")
# clusters, stats = find_connected_clusters_2d(data, adj, 1.96)
# ```
# """
function find_connected_clusters_2d(
    data::AbstractMatrix,
    adjacency::AbstractMatrix{Bool},
    threshold::Real,
)
    n_channels, n_times = size(data)
    
    # Validate adjacency matrix
    if size(adjacency) != (n_channels, n_channels)
        error("Adjacency matrix size ($(size(adjacency))) must match number of channels ($n_channels)")
    end
    
    # Find all samples exceeding threshold
    above_threshold = abs.(data) .>= threshold
    
    # Track which samples have been assigned to clusters
    visited = falses(size(data))
    
    clusters = Vector{Vector{CartesianIndex{2}}}()
    cluster_stats = Float64[]
    
    # Iterate through all positions
    for idx in CartesianIndices(data)
        # Skip if below threshold or already visited
        if !above_threshold[idx] || visited[idx]
            continue
        end
        
        # Start a new cluster
        cluster = CartesianIndex{2}[]
        queue = [idx]
        cluster_sum = 0.0
        
        while !isempty(queue)
            current = popfirst!(queue)
            
            # Skip if already visited
            if visited[current]
                continue
            end
            
            visited[current] = true
            push!(cluster, current)
            cluster_sum += abs(data[current])
            
            # Find neighbors (spatiotemporal connectivity)
            ch, t = Tuple(current)
            
            # Temporal neighbors (same channel, adjacent time points)
            for new_t in (t-1, t+1)
                if 1 <= new_t <= n_times
                    neighbor = CartesianIndex(ch, new_t)
                    if above_threshold[neighbor] && !visited[neighbor]
                        push!(queue, neighbor)
                    end
                end
            end
            
            # Spatial neighbors (adjacent channels, same time point)
            for new_ch in 1:n_channels
                if adjacency[ch, new_ch]  # Channels are spatial neighbors
                    neighbor = CartesianIndex(new_ch, t)
                    if above_threshold[neighbor] && !visited[neighbor]
                        push!(queue, neighbor)
                    end
                end
            end
        end
        
        # Add cluster if it contains any samples
        if !isempty(cluster)
            push!(clusters, cluster)
            push!(cluster_stats, cluster_sum)
        end
    end
    
    return clusters, cluster_stats
end


# """
#     spatiotemporal_cluster_pvalues(
#         rng::AbstractRNG,
#         observed::AbstractMatrix,
#         permuted::AbstractArray{<:Real,3},
#         adjacency::AbstractMatrix{Bool},
#         threshold::Real
#     )

# Compute p-values for spatiotemporal cluster-based permutation test.

# This function implements a cluster-based permutation test that accounts for 
# both spatial (across channels) and temporal (across time) structure in the data.

# # Arguments
# - `rng::AbstractRNG`: Random number generator for reproducibility
# - `observed::AbstractMatrix`: Observed test statistics (channels × time)
# - `permuted::AbstractArray{<:Real,3}`: Permuted test statistics (channels × time × n_permutations)
# - `adjacency::AbstractMatrix{Bool}`: Spatial adjacency matrix (channels × channels)
# - `threshold::Real`: Cluster-forming threshold

# # Returns
# - `pvalues::Matrix{Float64}`: P-values for each sample (channels × time)

# # Algorithm
# 1. Find clusters in observed data using spatiotemporal connectivity
# 2. For each permutation, find clusters and compute max cluster statistic
# 3. Build null distribution from max cluster statistics
# 4. Compute p-value for each observed cluster based on null distribution
# 5. Assign p-values to all samples within each cluster

# # Examples
# ```julia
# observed = modelfit_statistics  # channels × time
# permuted = permutation_statistics  # channels × time × n_perms
# adj, _ = read_fieldtrip_neighbors("biosemi64_neighb.mat")
# pvals = spatiotemporal_cluster_pvalues(rng, observed, permuted, adj, 1.96)
# ```
# """
function spatiotemporal_cluster_pvalues(
    rng::AbstractRNG,
    observed::AbstractMatrix,
    permuted::AbstractArray{<:Real,3},
    adjacency::AbstractMatrix{Bool},
    threshold::Real,
)
    n_channels, n_times = size(observed)
    n_permutations = size(permuted, 3)
    
    # Find clusters in observed data
    obs_clusters, obs_cluster_stats = find_connected_clusters_2d(observed, adjacency, threshold)
    
    # Build null distribution of maximum cluster statistics
    null_max_stats = zeros(n_permutations)
    
    for perm_idx in 1:n_permutations
        perm_data = permuted[:, :, perm_idx]
        _, perm_cluster_stats = find_connected_clusters_2d(perm_data, adjacency, threshold)
        
        if !isempty(perm_cluster_stats)
            null_max_stats[perm_idx] = maximum(perm_cluster_stats)
        else
            null_max_stats[perm_idx] = 0.0
        end
    end
    
    # Compute p-values for each observed cluster
    pvalues = ones(n_channels, n_times)  # Initialize with 1.0 (non-significant)
    
    for (cluster, cluster_stat) in zip(obs_clusters, obs_cluster_stats)
        # Count how many permutations had a max cluster stat >= this cluster's stat
        n_extreme = sum(null_max_stats .>= cluster_stat)
        cluster_pvalue = (n_extreme + 1) / (n_permutations + 1)
        
        # Assign p-value to all samples in this cluster
        for idx in cluster
            pvalues[idx] = cluster_pvalue
        end
    end
    
    return pvalues
end




"""
    lmm_permutations(rng::AbstractRNG,model::UnfoldLinearMixedModel,data::AbstractArray{<:Real,3},coefficient::Int;kwargs...)

Calculates permutations of a UnfoldLinearMixedModel object

# Arguments
- `rng::AbstractRNG`: An RNG-generator for reproducibility
- `model::UnfoldLinearMixedModel`: The fitted UnfoldLinearMixedModel to calculate the permutations of
- `data::AbstractArray{<:Real,3}`: The data with ch x time x trials
- `coefficient:Int`: The coefficient to test the null of

# Keyword arguments
- `n_permutations::Int = 500`: Number of permutations. Based on other permutation work, 500 should be a reasonable number. Methods based on e.g. tail-approximation are not yet available
- `lmm_statistic = :z`: What statistic to extract, could also be e.g. `β`
- `time_selection = 1:size(data, 2)`: Possibility to calculate permutations on a subset of time points

# Returns
- `result::Array` : Returns the full permutation Array with ch x timepoints x permutations for the coefficient-of-interest
"""
function lmm_permutations(
    rng::AbstractRNG,
    model,
    data::AbstractArray{<:Real,3},
    coefficient::Int;
    n_permutations = 500,
    lmm_statistic = :z,
    time_selection = 1:size(data, 2),
)
    permdata = Array{Float64}(undef, size(data, 1), length(time_selection), n_permutations)

    Xs = UnfoldMixedModels.prepare_modelmatrix(model)

    mm_outer = UnfoldMixedModels.LinearMixedModel_wrapper(
        Unfold.formulas(model),
        data[1, 1, :],
        Xs,
    )
    mm_outer.optsum.maxtime = 0.1 # 

    chIx = 1 # for now we only support 1 channel anyway
    #
    #p = Progress(length(time_selection))

    @assert(
        coefficient <= size(coef(mm_outer))[end],
        "chosen coefficient was larger than available coefficients"
    )
    #Threads.@threads for tIx =1:length(time_selection)
    #@showprogress "Processing Timepoints" for 
    for chIx = 1:size(data, 1)
        #Threads.@threads 
        for tIx = 1:length(time_selection)

            # splice in the correct dataa for residual calculation
            mm = deepcopy(mm_outer)
            mm.y .= data[chIx, time_selection[tIx], :]

            # set the previous calculated model-fit
            θ = Vector(modelfit(model).θ[time_selection[tIx]])
            @debug size(θ)
            MixedModels.updateL!(MixedModels.setθ!(mm, θ))

            # get the coefficient 
            H0 = coef(mm)
            # set the one of interest to 0
            H0[coefficient] = 0
            # run the permutation

            permutations = undef
            Logging.with_logger(NullLogger()) do   # remove NLopt warnings  
                permutations = permutation(
                    deepcopy(rng), # important here is to set the same seed to keep flip all time-points the same
                    n_permutations,
                    mm;
                    β = H0,
                    progress = false,
                    #blup_method = MixedModelsPermutations.olsranef,
                ) # constant rng to keep autocorr & olsranef for singular models
            end

            # extract the test-statistic

            permdata[chIx, tIx, :] =
                get_lmm_statistic(model, permutations, coefficient, lmm_statistic)

            #next!(p)
        end # end for
    end
    return permdata
end





"""
    get_lmm_statistic(model::UnfoldLinearMixedModel, coefficient::Int, lmm_statistic)
    get_lmm_statistic(model,permutations::MixedModelFitCollection, coefficient, lmm_statistic)
    
Returns the field `lmm_statistic` of the `coefpvalues` table output of the `permutations`-FitCollection only of the coefficient `coefnames(formulas(model))[1][coefficient]`

# Arguments
- `model`: typically an `UnfoldMixedModel` or similar
- `permutations::MixedModelFitCollection`: the output of `modelfit(model)`
- `coefficient::Int`: the coefficient to choose (a fixed effect)
- `lmm_statistic::Symbol`: The statistic to extact, tyically either `β` or `z`


# Returns
- `result::Vector` : A vector of the extracted `lmm_statistic`

"""
function get_lmm_statistic(
    model,
    permutations::MixedModelsPermutations.MixedModels.MixedModelFitCollection,
    coefficient::Int,
    lmm_statistic,
)
    [
        getproperty(m, lmm_statistic) for m in permutations.coefpvalues if
        String(m.coefname) == Unfold.coefnames(Unfold.formulas(model))[1][coefficient]
    ]

end
function get_lmm_statistic(model::UnfoldLinearMixedModel, coefficient::Int, lmm_statistic)
    return get_lmm_statistic(model, modelfit(model), coefficient, lmm_statistic)
    #    r = coeftable(m)
    #    r = subset(r, :group => (x -> isnothing.(x)), :coefname => (x -> x .!== "(Intercept)"))
    #    tvals = abs.(r.estimate ./ r.stderror)
    #    return tvals
end
