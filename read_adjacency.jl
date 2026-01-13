"""
    read_fieldtrip_neighbors(filepath::AbstractString; channel_order=nothing)

Read a FieldTrip/MNE neighbor definition file (`.mat` format) and convert it to an adjacency matrix.

# Arguments
- `filepath::AbstractString`: Path to the neighbor `.mat` file

# Keyword Arguments
- `channel_order::Union{Nothing,AbstractVector{<:AbstractString}} = nothing`: Optional vector 
  of channel names specifying the desired order of channels in the adjacency matrix. If provided,
  the adjacency matrix will be reordered to match your data's channel order. Channels not found
  in the neighbor file will have no neighbors.

# Returns
- `adjacency::Matrix{Bool}`: An N×N boolean adjacency matrix
- `channel_names::Vector{String}`: Vector of channel names in the order they appear in the matrix

# Example files
- `biosemi32_neighb.mat`,`easycap32ch-avg_neighb.mat`, `easycap64ch-avg_neighb.mat`
- `eeg1010_neighb.mat`, `elec1005_neighb.mat`

"""
function read_fieldtrip_neighbors(filepath::AbstractString; 
                                   channel_order::Union{Nothing,AbstractVector{<:AbstractString}} = nothing)
    if !isfile(filepath)
        error("File not found: $filepath")
    end
        
    data = Main.MAT.matread(filepath)
        
    neighbors_struct = data["neighbours"]
    labels_data = neighbors_struct["label"]
    neighblabels_data = neighbors_struct["neighblabel"]
 
    # channel names and neighbour lists
    file_channels = String.(vec(labels_data))
    file_neighblabels = [String.(vec(nl)) for nl in neighblabels_data]

    # Determine output channel order
    if channel_order === nothing
        out_channels = file_channels
    else
        out_channels = String.(channel_order)
    end
    
    n = length(out_channels)
    
    # Build index mapping: file channel name -> file index
    file_ch_idx = Dict(ch => i for (i, ch) in enumerate(file_channels))
    
    # Build index mapping: output channel name -> output index
    out_ch_idx = Dict(ch => i for (i, ch) in enumerate(out_channels))
    
    # Build adjacency matrix in output order
    adj = falses(n, n)
    
    for (i, ch) in enumerate(out_channels)
        # Find this channel in the file
        if !haskey(file_ch_idx, ch)
            # Channel not in neighbor file - leave with no neighbors
            continue
        end
        file_i = file_ch_idx[ch]
        
        # Get neighbors from file
        for neigh in file_neighblabels[file_i]
            if haskey(out_ch_idx, neigh)
                j = out_ch_idx[neigh]
                adj[i, j] = true
                adj[j, i] = true
            end
        end
    end
    
    return adj, out_channels
end
