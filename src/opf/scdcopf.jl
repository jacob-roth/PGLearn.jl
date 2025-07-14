#=
second stage
=#
const XI = Vector{OPFModel2{DCOPF2}}
function make_stage_2(data::PGLearn.OPFData, data2::PGLearn.OPFData2, solver=Ipopt.Optimizer)
  return PGLearn.build_opf(PGLearn.DCOPF2, data, data2, solver)
end
function solve_stage_2!(gV::AbstractVector, pg::AbstractVector, pg_model_2::PGLearn.OPFModel2{PGLearn.DCOPF2})
  set_parameter_value.(pg_model_2.model[:pgbar], pg);
  # solve
  # PGLearn.solve!(pg_model_2)
  optimize!(pg_model_2.model)
  # res = PGLearn.extract_result(pg_model_2)
  V = objective_value(pg_model_2.model)
  gV .= PGLearn.extract_result(pg_model_2)["dual"]["pgbar"]
  return V, gV
end
function solve_stage_2(pg::AbstractVector, pg_model_2::PGLearn.OPFModel2{PGLearn.DCOPF2})
  gV = similar(pg)
  solve_stage_2!(gV, pg, pg_model_2)
end

#=
scdcopf
=#
struct SCDCOPF <: AbstractFormulation end

# function set_optimal_start_values(model::Model)
#   # Ref: https://discourse.julialang.org/t/jump-model-warm-start-using-ipopt/92660/2
#   # Store a mapping of the variable primal solution
#   variable_primal = Dict(x => value(x) for x in all_variables(model))
#   # In the following, we loop through every constraint and store a mapping
#   # from the constraint index to a tuple containing the primal and dual
#   # solutions.
#   constraint_solution = Dict()
#   nlp_dual_start = nonlinear_dual_start_value(model)
#   for (F, S) in list_of_constraint_types(model)
#     # We add a try-catch here because some constraint types might not
#     # support getting the primal or dual solution.
#     try
#       for ci in all_constraints(model, F, S)
#         constraint_solution[ci] = (value(ci), dual(ci))
#       end
#     catch
#       @info("Something went wrong getting $F-in-$S. Skipping")
#     end
#   end
#   # Now we can loop through our cached solutions and set the starting values.
#   for (x, primal_start) in variable_primal
#     set_start_value(x, primal_start)
#   end
#   for (ci, (primal_start, dual_start)) in constraint_solution
#     # set_start_value(ci, primal_start)
#     set_dual_start_value(ci, dual_start)
#   end
#   set_nonlinear_dual_start_value(model, nlp_dual_start)
#   return
# end
function set_optimal_start_values(model::Model)
  # primal
  pgbar = value.(model[:pg])
  pfbar = value.(model[:pf])
  ptbar = value.(model[:pt])
  vabar = value.(model[:va])
  # dual
  kclbar = dual.(model[:kcl_p])
  ohmbar = dual.(model[:ohm_pf])
  vadbar = dual.(model[:va_diff])
  # primal
  for (i,pg) in enumerate(model[:pg]);       set_start_value(pg, pgbar[i]);   end
  for (i,pf) in enumerate(model[:pf]);       set_start_value(pf, pfbar[i]);   end
  for (i,pt) in enumerate(model[:pt]);       set_start_value(-pt, -ptbar[i]);   end
  for (i,va) in enumerate(model[:va]);       set_start_value(va, vabar[i]);   end
  # dual
  for (i,kcl) in enumerate(model[:kcl_p]);   set_dual_start_value(kcl, kclbar[i]); end
  for (i,ohm) in enumerate(model[:ohm_pf]);  set_dual_start_value(ohm, ohmbar[i]); end
  for (i,vad) in enumerate(model[:va_diff]); set_dual_start_value(vad, vadbar[i]); end
end

function build_scopf(
  ::Type{SCDCOPF},
  data::OPFData,
  Xi::XI,
  p::Vector{<:Real},
  alpha::Real,
  c::Real,
  optimizer;
  warmstart::Bool=true,
  T=Float64,
)
  # base case
  opf = build_opf(DCOPF, data, optimizer)
  model = opf.model
  if warmstart
    optimize!(model)
    x_last = value.(model[:pg])
    set_optimal_start_values(model)
  else
    x_last = zeros(length(model[:pg]))
  end
  n = length(model[:pg])

  # second stage
  m = length(Xi)
  order_cached = 1
  V_cache = zeros(m)
  JV_cache = zeros(m, n)
  HV_cache = nothing
  VO = VectorOracle(m, n, Ref(order_cached), x_last, V_cache, JV_cache, HV_cache)
  @eval begin
  function CVaRUtilities.update!(VO::VectorOracle, pgbar::AbstractVector, Xi::XI; order::Int=1, force::Bool=false)
    if (pgbar != VO.x_last) || force
      for i in 1:VO.m
        @views VO.V_cache[i] = first(solve_stage_2!(VO.JV_cache[i,:], pgbar, Xi[i]))
      end
      # VO.order_cached[] = order
      VO.order_cached[] = 1
      VO.x_last .= pgbar
    end
    nothing
  end; end # eval

  C = SmoothCVaR(c, alpha, p, true)
  implicit_H = false
  sparse_H = false
  CVO = SmoothCVaR_V_Oracle(C, VO, implicit_H, sparse_H)

  function _C_V_val(x...)
    xvec = collect(x)
    v = CVaRUtilities.update!(CVO, xvec, Xi; order=0, force=false)
    return v
  end
  function _C_V_grad(g::AbstractVector, x...)
    xvec = collect(x)
    CVaRUtilities.update!(CVO, xvec, Xi; order=1, force=false)
    g .= CVO.g_cache
    return
  end
  function _C_V_hess(H::AbstractMatrix, x...)
    xvec = collect(x)
    CVaRUtilities.update!(CVO, xvec, Xi; order=2, force=false)
    for j in 1:n, i in j:n # lower triangle
      H[i,j] = CVO.H_cache[i,j]
    end
    return
  end
  JuMP.register(model, :C_V, n, _C_V_val, _C_V_grad, _C_V_hess)
  obj_old = objective_function(model)
  @NLobjective(model, Min, obj_old + C_V(model[:pg]...))
  return SCOPFModel{SCDCOPF}(opf.data, Xi, opf.model, CVO)
end
solve!(opf::SCOPFModel{SCDCOPF}) = optimize!(opf.model)

#=
deterministic equivalent
=#

struct SCDCOPF_DE <: AbstractFormulation end

"""
  build_opf(::Type{TwoStageDCOPF},
    data::OPFData,              # first-stage network data
    Xi::XI,                     # scenario-specific data
    p::Vector{<:Real},          # probabilities p[ω],  sum p = 1
    alpha::Real,                # cvar confidence: 1=max, 0=expectation
    optimizer; T = Float64)
"""
function build_scopf(
  ::Type{SCDCOPF_DE},
  data::OPFData,
  Xi::XI,
  p::Vector{Tp},
  alpha::Real,
  optimizer; T=Float64) where {Tp<:Real}
  # initialize
  @assert length(Xi) == length(p)  "length(Xi) ≠ length(p)"
  @assert isapprox(sum(p), one(Tp); atol = 1e-8)  "probabilities must sum to 1"
  model = JuMP.GenericModel{T}(optimizer)
  model.ext[:opf_model] = SCDCOPF_DE

  # -------------------------------------------------------------
  # stage 1
  # -------------------------------------------------------------
  
  # data
  N_1, E_1, G_1 = data.N, data.E, data.G
  gs_1 = data.gs
  pd_1 = data.pd
  i0_1 = data.ref_bus
  bus_arcs_fr_1, bus_arcs_to_1 = data.bus_arcs_fr, data.bus_arcs_to
  bus_gens_1 = data.bus_gens
  pgmin_1, pgmax_1 = data.pgmin, data.pgmax
  c0_1, c1_1, c2_1 = data.c0, data.c1, data.c2
  gen_status_1 = data.gen_status
  bus_fr_1, bus_to_1, b_1 = data.bus_fr, data.bus_to, data.b
  dvamin_1, dvamax_1, smax_1 = data.dvamin, data.dvamax, data.smax
  branch_status_1 = data.branch_status
  
  #
  # Ia. first-stage variables (shared across all scenarios)
  #

  # nodal voltage
  @variable(model, va_1[1:N_1])

  # Active and reactive dispatch
  @variable(model, pg_1[1:G_1])

  # Branch flows
  @variable(model, pf_1[1:E_1])

  # 
  # Ib. constraints
  #

  # Generation bounds (both zero if generator is off)
  set_lower_bound.(pg_1, gen_status_1 .* pgmin_1)
  set_upper_bound.(pg_1, gen_status_1 .* pgmax_1)

  # Flow bounds (both zero if branch is off)
  set_lower_bound.(pf_1, branch_status_1 .* -smax_1)
  set_upper_bound.(pf_1, branch_status_1 .* smax_1)

  # Slack bus
  @constraint(model, slack_bus_1, va_1[i0_1] == 0.0)

  # Nodal power balance
  @expression(model, pt_1[e in 1:E_1], -pf_1[e])
  @constraint(model,
      kcl_p_1[i in 1:N_1],
      sum(gen_status_1[g] * pg_1[g] for g in bus_gens_1[i])
      - sum(branch_status_1[a] * pf_1[a] for a in bus_arcs_fr_1[i])
      - sum(branch_status_1[a] * pt_1[a] for a in bus_arcs_to_1[i])  # pt == -pf
      == 
      sum(pd_1[l] for l in data.bus_loads[i]) + gs_1[i]
  )

  @constraint(model,
      ohm_pf_1[e in 1:E_1],
      branch_status_1[e] * (
          -b_1[e] * (va_1[bus_fr_1[e]] - va_1[bus_to_1[e]])
      ) - pf_1[e] == 0
  )
  @constraint(model,
      va_diff_1[e in 1:E_1],
      dvamin_1[e] ≤ branch_status_1[e] * (va_1[bus_fr_1[e]] - va_1[bus_to_1[e]]) ≤ dvamax_1[e]
  )

  # -------------------------------------------------------------
  # stage 2
  # -------------------------------------------------------------

  S = length(Xi) # number of scenarios
  @variable(model, second_stage_cost[1:S])

  for s in 1:S
    # data
    xi = Xi[s]

    ramp_frac_2  = xi.data2.ramp_frac
    c_up_scale_2 = xi.data2.c_up_scale
    c_dn_scale_2 = xi.data2.c_dn_scale
    c_sh_2       = xi.data2.c_sh

    N_2, E_2, G_2 = xi.data.N, xi.data.E, xi.data.G
    L_2 = length(xi.data.pd)
    gs_2 = xi.data.gs
    pd_2 = xi.data.pd
    i0_2 = xi.data.ref_bus
    bus_arcs_fr_2, bus_arcs_to_2 = xi.data.bus_arcs_fr, xi.data.bus_arcs_to
    bus_gens_2 = xi.data.bus_gens
    pgmin_2, pgmax_2 = xi.data.pgmin, xi.data.pgmax
    c0_2, c1_2, c2_2 = xi.data.c0, xi.data.c1, xi.data.c2
    gen_status_2 = xi.data.gen_status
    bus_fr_2, bus_to_2, b_2 = xi.data.bus_fr, xi.data.bus_to, xi.data.b
    dvamin_2, dvamax_2, smax_2 = xi.data.dvamin, xi.data.dvamax, xi.data.smax
    branch_status_2 = xi.data.branch_status

    #
    # IIa. second-stage variables
    #

    # nodal voltage
    va_2 = @variable(model, [i=1:N_2], base_name = "va_2_$(s)")
    model[Symbol("va_2_",s)] = va_2

    # Active and reactive dispatch
    pg_2 = @variable(model, [g = 1:G_2], base_name = "pg_2_$(s)")
    model[Symbol("pg_2_",s)] = pg_2
    delta_pg_pos_2 = @variable(model,
      [g in 1:G_2],
      lower_bound = 0,
      upper_bound = gen_status_2[g] * ramp_frac_2 * (pgmin_2[g] + pgmax_2[g]),
      base_name="delta_pg_pos_2_$(s)"
    )
    model[Symbol("delta_pg_pos_2_",s)] = delta_pg_pos_2
    delta_pg_neg_2 = @variable(model,
      [g in 1:G_2],
      lower_bound = 0,
      upper_bound = gen_status_2[g] * ramp_frac_2 * (pgmin_2[g] + pgmax_2[g]),
      base_name="delta_pg_neg_2_$(s)"
    )
    model[Symbol("delta_pg_neg_2_",s)] = delta_pg_neg_2
    redispatch_2 = @constraint(model,
      [g in 1:xi.data.G],
      pg_2[g] == pg_1[g] + delta_pg_pos_2[g] - delta_pg_neg_2[g]
    )
    model[Symbol("redispatch_2_",s)] = redispatch_2

    # Branch flows
    pf_2 = @variable(model, [e=1:E_2], base_name = "pf_2_$(s)")
    model[Symbol("pf_2_",s)] = pf_2
    
    # load shed
    shed_2 = @variable(model,
      [l=1:L_2],
      lower_bound = 0,
      upper_bound = xi.data.pd[l],
      base_name = "shed_2_$(s)"
    )
    model[Symbol("shed_2_",s)] = shed_2
    
    #
    # IIb. second-stage constraints
    #

    # Generation bounds (both zero if generator is off)
    set_lower_bound.(pg_2, gen_status_2 .* pgmin_2)
    set_upper_bound.(pg_2, gen_status_2 .* pgmax_2)

    # Flow bounds (both zero if branch is off)
    set_lower_bound.(pf_2, branch_status_2 .* -smax_2)
    set_upper_bound.(pf_2, branch_status_2 .* smax_2)

    # Slack bus
    slack_bus_2 = @constraint(model, va_2[i0_2] == 0.0)
    model[Symbol("slack_bus_2_",s)] = slack_bus_2

    # Nodal power balance
    pt_2 = @expression(model, [e in 1:E_2], -pf_2[e])
    model[Symbol("pt_2_",s)] = pt_2
    kcl_p_2 = @constraint(model,
        [i in 1:N_2],
        sum(gen_status_2[g] * pg_2[g] for g in bus_gens_2[i])
        - sum(branch_status_2[a] * pf_2[a] for a in bus_arcs_fr_2[i])
        - sum(branch_status_2[a] * pt_2[a] for a in bus_arcs_to_2[i])  # pt == -pf
        == 
        sum(pd_2[l] - shed_2[l] for l in xi.data.bus_loads[i]) + gs_2[i]
    )
    model[Symbol("kcl_p_2_",s)] = kcl_p_2

    ohm_pf_2 = @constraint(model, 
        [e in 1:E_2],
        branch_status_2[e] * (
            -b_2[e] * (va_2[bus_fr_2[e]] - va_2[bus_to_2[e]])
        ) - pf_2[e] == 0
    )
    model[Symbol("ohm_pf_2_",s)] = ohm_pf_2
    va_diff_2 = @constraint(model,
        [e in 1:E_2],
        dvamin_2[e] ≤ branch_status_2[e] * (va_2[bus_fr_2[e]] - va_2[bus_to_2[e]]) ≤ dvamax_2[e]
    )
    model[Symbol("va_diff_2_",s)] = va_diff_2

    #
    # IIc. second-stage cost
    #
    l, u = extrema(c2_2)
    (l == u == 0.0) || @warn "Data $(data.case) has quadratic cost terms; those terms are being ignored"

    # incremental dispatch cost
    @constraint(model, second_stage_cost[s] >=
      sum(
              c_up_scale_2 * c1_2[g] * delta_pg_pos_2[g] + c_dn_scale_2 * c1_2[g] * delta_pg_neg_2[g] + c0_2[g]
              for g in 1:G_2 if gen_status_2[g]
          ) + sum(
              sum(c_sh_2 * shed_2[l] for l in xi.data.bus_loads[i])
              for i in 1:N_2
          )
    )
  end

  #
  # III. objective
  #
  
  total_first_stage_cost = sum(c1_1[g] * pg_1[g] + c0_1[g] for g in 1:G_1 if gen_status_1[g])
  
  @variable(model, z[i=1:S] >= 0)
  @variable(model, tau)
  z_cost = @constraint(model, z .>= second_stage_cost .- tau)
  total_second_stage_cost = tau + sum(p .* z)/(1-alpha)
  
  @objective(model, Min, total_first_stage_cost + total_second_stage_cost)
  
  return SCOPFModel{SCDCOPF_DE}(data, Xi, model, nothing)
end
solve!(opf::SCOPFModel{SCDCOPF_DE}) = optimize!(opf.model)


extract_primal(opf::SCOPFModel{SCDCOPF}) = extract_primal(OPFModel{DCOPF}(opf.data, opf.model))
extract_dual(opf::SCOPFModel{SCDCOPF}) = extract_dual(OPFModel{DCOPF}(opf.data, opf.model))

"""
    extract_primal(opf::SCOPFModel{SCDCOPF_DE})

Return a Dict with the first-stage primal variables and a nested Dict
`"scenarios"` that maps each scenario index `s` → its own Dict of
second-stage variables.
"""
function extract_primal(opf::SCOPFModel{SCDCOPF_DE})
    model = opf.model
    data  = opf.data
    Xi    = opf.xi_data
    S     = length(Xi)

    T = JuMP.value_type(typeof(model))

    # -------------------- stage-1 --------------------
    prim = Dict{String,Any}()

    if has_values(model)
        prim["va_1"] = value.(model[:va_1])
        prim["pg_1"] = value.(model[:pg_1])
        prim["pf_1"] = value.(model[:pf_1])
        prim["z"] = value.(model[:z])
        prim["tau"] = value(model[:tau])
        prim["second_stage_cost"] = value.(model[:second_stage_cost])
    else
        prim["va_1"] = zeros(T, data.N)
        prim["pg_1"] = zeros(T, data.G)
        prim["pf_1"] = zeros(T, data.E)
        prim["z"] = zeros(T, S)
        prim["tau"] = zero(T)
        prim["second_stage_cost"] = zeros(T, S)
    end

    # -------------------- stage-2 (per scenario) -----
    scen = Dict{Int, Dict{String,Any}}()

    for s in 1:S
        key = Dict{String,Any}()

        va   = model[Symbol("va_2_",   s)]
        pg   = model[Symbol("pg_2_",   s)]
        dpos = model[Symbol("delta_pg_pos_2_", s)]
        dneg = model[Symbol("delta_pg_neg_2_", s)]
        pf   = model[Symbol("pf_2_",   s)]
        shed = model[Symbol("shed_2_", s)]   # <-- adjust if key changed

        if has_values(model)
            key["va_2"]          = value.(va)
            key["pg_2"]          = value.(pg)
            key["delta_pg_pos_2"] = value.(dpos)
            key["delta_pg_neg_2"] = value.(dneg)
            key["pf_2"]          = value.(pf)
            key["shed_2"]        = value.(shed)
        else
            key["va_2"]          = zeros(T, length(va))
            key["pg_2"]          = zeros(T, length(pg))
            key["delta_pg_pos_2"] = zeros(T, length(dpos))
            key["delta_pg_neg_2"] = zeros(T, length(dneg))
            key["pf_2"]          = zeros(T, length(pf))
            key["shed_2"]        = zeros(T, length(shed))
        end
        scen[s] = key
    end

    prim["scenarios"] = scen
    return prim
end



"""
    extract_dual(opf::SCOPFModel{SCDCOPF_DE})

Analogous to `extract_primal` but collects constraint duals and
bound-multipliers.
"""
function extract_dual(opf::SCOPFModel{SCDCOPF_DE})
    model = opf.model
    data  = opf.data
    Xi    = opf.xi_data
    S     = length(Xi)

    T = JuMP.value_type(typeof(model))

    duals = Dict{String,Any}()

    # ------------- stage-1 -------------
    if has_duals(model)
        duals["slack_bus_1"] = dual(model[:slack_bus_1])
        duals["kcl_p_1"]     = dual.(model[:kcl_p_1])
        duals["ohm_pf_1"]    = dual.(model[:ohm_pf_1])
        duals["va_diff_1"]   = dual.(model[:va_diff_1])

        duals["pg_1"] = dual.(LowerBoundRef.(model[:pg_1])) +
                        dual.(UpperBoundRef.(model[:pg_1]))
        duals["pf_1"] = dual.(LowerBoundRef.(model[:pf_1])) +
                        dual.(UpperBoundRef.(model[:pf_1]))
    else
        duals["slack_bus_1"] = zero(T)
        duals["kcl_p_1"]     = zeros(T, data.N)
        duals["ohm_pf_1"]    = zeros(T, data.E)
        duals["va_diff_1"]   = zeros(T, data.E)
        duals["pg_1"]        = zeros(T, data.G)
        duals["pf_1"]        = zeros(T, data.E)
    end

    # ------------- stage-2 per scenario -------------
    scen_dual = Dict{Int, Dict{String,Any}}()

    for s in 1:S
        d = Dict{String,Any}()

        kcl   = model[Symbol("kcl_p_2_",   s)]
        ohm   = model[Symbol("ohm_pf_2_",  s)]
        vdiff = model[Symbol("va_diff_2_", s)]
        slk   = model[Symbol("slack_bus_2_", s)]

        if has_duals(model)
            d["slack_bus_2"] = dual(slk)
            d["kcl_p_2"]     = dual.(kcl)
            d["ohm_pf_2"]    = dual.(ohm)
            d["va_diff_2"]   = dual.(vdiff)

            pg   = model[Symbol("pg_2_",   s)]
            dpos = model[Symbol("delta_pg_pos_2_", s)]
            dneg = model[Symbol("delta_pg_neg_2_", s)]
            pf   = model[Symbol("pf_2_",   s)]
            shed = model[Symbol("shed_2_", s)]   # <-- adjust if key changed

            d["pg_2"]          = dual.(LowerBoundRef.(pg))   +
                                 dual.(UpperBoundRef.(pg))
            d["delta_pg_pos_2"] = dual.(UpperBoundRef.(dpos)) # only upper bound
            d["delta_pg_neg_2"] = dual.(UpperBoundRef.(dneg))
            d["pf_2"]          = dual.(LowerBoundRef.(pf))   +
                                 dual.(UpperBoundRef.(pf))
            d["shed_2"]        = dual.(UpperBoundRef.(shed))
        else
            # same shapes, filled with zeros
            d["slack_bus_2"] = zero(T)
            d["kcl_p_2"]     = zeros(T, length(kcl))
            d["ohm_pf_2"]    = zeros(T, length(ohm))
            d["va_diff_2"]   = zeros(T, length(vdiff))

            pg   = model[Symbol("pg_2_",   s)]
            dpos = model[Symbol("delta_pg_pos_2_", s)]
            dneg = model[Symbol("delta_pg_neg_2_", s)]
            pf   = model[Symbol("pf_2_",   s)]
            shed = model[Symbol("shed_2_", s)]   # <-- adjust if key changed

            d["pg_2"]          = zeros(T, length(pg))
            d["delta_pg_pos_2"] = zeros(T, length(dpos))
            d["delta_pg_neg_2"] = zeros(T, length(dneg))
            d["pf_2"]          = zeros(T, length(pf))
            d["shed_2"]        = zeros(T, length(shed))
        end
        scen_dual[s] = d
    end

    duals["scenarios"] = scen_dual
    return duals
end
