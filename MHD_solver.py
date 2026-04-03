# AUTHOR: Enrico Zampa, University of Vienna

import argparse
import numpy as np
from ngsolve import *

# ------------------------------------------------------------------------------
# 1. SETUP ARGUMENT PARSER
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NGSolve Simulation with Overwrite Params")

# Define optional arguments. Defaults are set to None so we can detect if they were passed.
parser.add_argument("--ictype", type=int, default=None, help="Initial Condition Type")
parser.add_argument("--NMAX", type=int, default=None, help="Max Mesh Scale Factor")
parser.add_argument("--order", type=int, default=None, help="FEM Order")
parser.add_argument("--dt", type=float, default=None, help="Time Step")
parser.add_argument("--method_type", type=int, default=None, help="Method Type")
parser.add_argument("--nu", type=float, default=None, help="Viscosity (overwrites mu)")
parser.add_argument("--eta", type=float, default=None, help="Resistivity")
parser.add_argument("--tend", type=float, default=None, help="End Time")
parser.add_argument("--Bx0", type=float, default = None, help="Initial Bx for LDC")
parser.add_argument("--By0", type=float, default = None, help="Initial By for LDC")
parser.add_argument("--nplot", type = int, default = None, help="Plot every nplot" )

try:
    args = parser.parse_args()
except SystemExit:
    # Fallback for Jupyter notebooks if args are empty or conflict
    args = parser.parse_args(args=[])

# ------------------------------------------------------------------------------
# 2. INITIALIZE GLOBAL PARAMETERS (With Overwrite Logic)
# ------------------------------------------------------------------------------

# Default values (The "Existing Ones")
default_order = 1
default_gammaNitsche = 10
default_ictype = 1
default_NMAX = 10
default_method_type = 3 
default_dt = 0.1

# Apply Overwrite: If args.val is present, use it; otherwise use default.
order = args.order if args.order is not None else default_order
ictype = args.ictype if args.ictype is not None else default_ictype
NMAX = args.NMAX if args.NMAX is not None else default_NMAX
method_type = args.method_type if args.method_type is not None else default_method_type


# Fixed parameters
gammaNitsche = default_gammaNitsche
nplot = 10
ubnd = CF((0,0))
f = CF((0,0))
g = CF((0,0))
lag_mult = False
time = Parameter(0.0)
Ebnd = CF(0)

def cross2d(a,b):
    return a[0]*b[1] - a[1]*b[0]

def my_max(a,b):
    return IfPos(a-b, a, b)


if method_type == 0:
    method_name = "_method0_"
    gammaCIPu = 0
    gammaCIPu_strong = 0
    gammaCIPgradu = 0
    gammaCIPcurlB = 0
    gammaCIPB = 0
elif method_type == 1:
    method_name = "_method1_"

    gammaCIPu = 0
    gammaCIPu_strong = 0.1
    gammaCIPgradu = 0
    gammaCIPcurlB = 0
    gammaCIPB = 0
elif method_type == 2:
    method_name = "_method2_"

    lag_mult = True
    gammaCIPu = 0
    gammaCIPu_strong = 0.1
    gammaCIPgradu = 0
    gammaCIPcurlB = 0
    gammaCIPB = 0.1
elif method_type == 3:
    lag_mult = False
    method_name = "_method3_"

    gammaCIPu = 0.1
    gammaCIPu_strong = 0
    gammaCIPgradu = 0.025
    gammaCIPcurlB = 0.025
    gammaCIPB = 0
elif method_type == 4:
    method_name = "_method4_"

    # method_1 + jumps of derivatives 
    gammaCIPu = 0
    gammaCIPu_strong = 0.1
    gammaCIPgradu = 0.025
    gammaCIPcurlB = 0.025
    gammaCIPB = 0

# CREATE PERIODIC MESH
from netgen.occ import *
from netgen.geom2d import SplineGeometry
from netgen.meshing import IdentificationType

if args.nu is not None:
    mu = args.nu
    print(f"Overwriting mu -> {mu}")

if args.eta is not None:
    eta = args.eta
    print(f"Overwriting eta -> {eta}")

if args.nplot is not None:
    nplot = args.nplot
    print(f"Overwriting nplot -> {nplot}")

scaleFactor = 1
if ictype == 1:
    tend = 0.1
    icname = "manufactured_sol"
    periodic_x = False
    periodic_y = False

    expfun = CF( -exp(-0.5*time))
    ddt_expfun = CF( 0.5*exp(-0.5*time))
    psifun = CF(sin(pi*x)**2 * sin(pi*y)**2)
    uspace= CF((psifun.Diff(y), -psifun.Diff(x)))  # u with homogeneous Dirichlet BC
    #uspace = CF((sin(2*pi*x)*sin(2*pi*y), cos(2*pi*x)*cos(2*pi*y)))
    psiB = CF( sin(pi*x)*sin( pi*y) )
    Bspace =CF( ( psiB.Diff(y), -psiB.Diff(x)))
    #Bspace = CF((sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(pi*y)))
    u0 = -uspace # initial condition
    B0 = -Bspace # initial condition
    ufun = expfun * uspace
    ubnd = ufun
    curlufun = CF( -ufun[0].Diff(y) + ufun[1].Diff(x) )
    curlcurlufun = CF( (curlufun.Diff(y),  - curlufun.Diff(x)) )
    Bfun = expfun * Bspace
    Bbnd = Bfun
    p0 = CF(sin(2*pi*x)*cos(2*pi*y))
    pfun = expfun * p0
    curlBfun = CF(-Bfun[0].Diff(y) + Bfun[1].Diff(x) )
    curlcurlBfun = CF( (curlBfun.Diff(y), - curlBfun.Diff(x) ))
    Btimesu = cross2d(Bfun, ufun)
    e1 = [1 , 0]
    e2 = [0, 1]
    adv = CF((cross2d(ufun, e1)*curlufun, cross2d(ufun, e2)*curlufun) ) #
    lorentz = CF((cross2d(Bfun, e1)*curlBfun, cross2d(Bfun, e2)*curlBfun) )
    advB = CF( (Btimesu.Diff(y), - Btimesu.Diff(x)) )
    gradp = CF((pfun.Diff(x), pfun.Diff(y)))
    f = CF( ddt_expfun * uspace + adv + mu*curlcurlufun - lorentz + gradp )
    g = CF( ddt_expfun * Bspace  + advB + eta*curlcurlBfun )
    Ebnd =  Btimesu + eta*curlBfun
    scaleFactor = 1
    shape = unit_square_shape.Move((0,0, 0) ).Scale((0,0,0),scaleFactor)
elif ictype == 2:
    scaleFactor = 1
    icname = "OT"
    tend = 0.8
    shape = unit_square_shape
    u0 = CF((- sin(2*pi*y), sin(2*pi*x)))
    B0 = CF((-sin(2*pi*y), sin(4*pi*x) ))

    periodic_x = True
    periodic_y = True
elif ictype == 3:
    scaleFactor = 1
    icname = "LDC"
    tend = 20
    ulid = 4*(1-x)*x
    ubnd = CF((IfPos(y-0.9999, ulid, 0), 0))
    shape = unit_square_shape.Move((0, 0, 0) ).Scale((0,0,0),scaleFactor)
    u0 = CF((0,0))
    Bx0 = args.Bx0 if args.Bx0 is not None else 0.
    By0 = args.By0 if args.By0 is not None else 0.
    B0 = CF((Bx0,By0))
    Bbnd = B0
    curlBfun = CF(0)

    periodic_x = False
    periodic_y = False

elif ictype == 4:
    scaleFactor = 2*pi
    icname = "VROT"
    tend = 2
    shape = unit_square_shape.Move((0, 0, 0) ).Scale((0,0,0),scaleFactor)
    uaux = sqrt(4*pi)
    u0 = CF((- uaux*sin(y), uaux * sin(x)))
    B0 = CF((-sin(y), sin(2*x) ))

    periodic_x = True
    periodic_y = True

elif ictype == 5:
    icname = "LoopAdvection"
    # loop advection
    shape = unit_square_shape
    tend = 1

    u0 = CF((1,1))
    r = sqrt((x-0.5)**2 + (y -0.5)**2)
    A0 = IfPos(r - 0.3 , 0, 0.001*(0.3 - r) )
    B0 = CF((A0.Diff(y), -A0.Diff(x)))

    periodic_x = True
    periodic_y = True
elif ictype == 6:
    icname = "MHDvortex"
    # MHD vortex
    scaleFactor = 10
    shape = unit_square_shape.Move((0, 0, 0) ).Scale((0,0,0),scaleFactor)
    tend = 0.1
    r = sqrt((x-5)**2 + (y-5)**2)
    uaux = exp(0.5*(1-r**2))
    u0 = CF( (-uaux *(y-5),uaux*(x-5) ))
    B0 = CF( (-uaux *(y-5),uaux*(x-5) ))

    mu = 0
    eta = 0

    Bbnd = B0
    ubnd = u0

    ufun = u0
    Bfun = B0

    periodic_x = False
    periodic_y = False

elif ictype == 7: 
    # nonsmooth solution 

    tend = 0.1
    icname = "Lshape"
    periodic_x = False
    periodic_y = False

    expfun = 1
    ddt_expfun = 0
    psifun = CF(sin(pi*x)**2 * sin(pi*y)**2)
    uspace= CF((psifun.Diff(y), -psifun.Diff(x)))  # u with homogeneous Dirichlet BC
    r = sqrt(x**2 + y**2)
    theta = atan2(y, x)
    psiB  = pow(r, 2/3) * sin((2/3)*theta)   
    Bspace = CF( ( psiB.Diff(x), psiB.Diff(y)))
    u0 = 1 * uspace # initial condition
    B0 = 1 * Bspace # initial condition
    ufun = expfun * uspace
    ubnd = ufun
    curlufun = CF( -ufun[0].Diff(y) + ufun[1].Diff(x) )
    curlcurlufun = CF( (curlufun.Diff(y),  - curlufun.Diff(x)) )
    Bfun = expfun * Bspace
    Bbnd = Bfun
    p0 = CF(0)
    pfun = expfun * p0
    curlBfun = CF(-Bfun[0].Diff(y) + Bfun[1].Diff(x) )
    curlcurlBfun = CF( (curlBfun.Diff(y), - curlBfun.Diff(x) ))
    Btimesu = cross2d(Bfun, ufun)
    e1 = [1 , 0]
    e2 = [0, 1]
    adv = CF((cross2d(ufun, e1)*curlufun, cross2d(ufun, e2)*curlufun) ) #
    lorentz = CF((cross2d(Bfun, e1)*curlBfun, cross2d(Bfun, e2)*curlBfun) )
    advB = CF( (Btimesu.Diff(y), - Btimesu.Diff(x)) )
    gradp = CF((pfun.Diff(x), pfun.Diff(y)))
    f = CF( ddt_expfun * uspace + adv + mu*curlcurlufun - lorentz + gradp )
    g = CF( ddt_expfun * Bspace  + advB + eta*curlcurlBfun )
    Ebnd =  Btimesu + eta*curlBfun
    scaleFactor = 1
    pnts =[(0,0),
       (0,-1),
       (1,-1),
       (1,1),
       (-1,1),
       (-1,0)]
    geo = SplineGeometry()
    pts = [geo.AppendPoint(*pnt) for pnt in pnts]
    curves = []
    curves.append([["line", pts[0], pts[1]], "dirichlet"])
    curves.append([["line", pts[1], pts[2]], "dirichlet"])
    curves.append([["line", pts[2], pts[3]], "dirichlet"])
    curves.append([["line", pts[3], pts[4]], "dirichlet"])
    curves.append([["line", pts[4], pts[5]], "dirichlet"])
    curves.append([["line", pts[5], pts[0]], "dirichlet"])
    [geo.Append(c,bc=bc) for c,bc in curves]
elif ictype == 8:
    scaleFactor = 1
    icname = "LDC_hard"
    tend = 20
    ulid = 1
    ubnd = CF((IfPos(y-0.9999, ulid, 0), 0))
    shape = unit_square_shape.Move((0, 0, 0) ).Scale((0,0,0),scaleFactor)
    u0 = CF((0,0))
    Bx0 = args.Bx0 if args.Bx0 is not None else 0.
    By0 = args.By0 if args.By0 is not None else 0.
    B0 = CF((Bx0,By0))
    Bbnd = B0
    curlBfun = CF(0)

    periodic_x = False
    periodic_y = False



# ------------------------------------------------------------------------------
# 3. OVERWRITE DATA HERE (Physics Parameters)
# ------------------------------------------------------------------------------

dtCFL = 0.1* (1./NMAX)**((order + 1)/2 )
nsteps = np.ceil( tend  /  dtCFL )
dtCFL = tend / nsteps
dt = args.dt if args.dt is not None else dtCFL


# Print config for verification
print(f"--- Configuration ---")
print(f"ictype: {ictype}, NMAX: {NMAX}, order: {order}, dt: {dt}, method: {method_type}")

# The code blocks above set defaults for mu, eta, and tend based on ictype.
# We overwrite them now ONLY if the user explicitly provided arguments.


if args.tend is not None:
    tend = args.tend
    print(f"Overwriting tend -> {tend}")
# ------------------------------------------------------------------------------

hmax = scaleFactor/NMAX

if ictype == 7:
    ngmesh = geo.GenerateMesh(maxh=hmax)
    mesh = Mesh(ngmesh)
else:


    shape.edges.Max(X).name = "right"
    shape.edges.Min(X).name = "left"
    shape.edges.Max(Y).name = "top"
    shape.edges.Min(Y).name = "bottom"

    if (periodic_y):
        print("Indentify top and bottom boundary")
        shape.edges.Max(Y).Identify(shape.edges.Min(Y), "bt")
    if (periodic_x):
        print("Identify left and right boundary")
        shape.edges.Max(X).Identify(shape.edges.Min(X), "lr")

    geo = OCCGeometry(shape, dim = 2)

    
   

    mesh = Mesh(geo.GenerateMesh(maxh=hmax))


C_S= 0.1 # 

fescurl = Periodic( HCurl(mesh, order = order) )
fesp = Periodic( H1(mesh, order = order + 1) )
N = NumberSpace(mesh)

if lag_mult:
    fes = FESpace([fescurl, fesp, fescurl, fesp, N, N], dgjumps = True)
    (u, p, B, phi, rho, zeta), (v, q, C, psi, drho, dzeta) = fes.TnT()

    gf = GridFunction(fes)
    gfu, gfp, gfB, gfphi, gfrho, gfzeta = gf.components

    gf_old = GridFunction(fes)
    gfu_old, gfp_old, gfB_old, gfphi_old, gfrho_old,  gfzeta= gf_old.components

    gf_star = GridFunction(fes)
    gfu_star, gfp_star, gfB_star, gfphi_star, gfrho_star,  gfzeta = gf_star.components

    gf_ex = GridFunction(fes)
    gfu_ex, gfp_ex, gfB_ex, gfphi_ex, gfrho_ex,  gfzeta = gf_ex.components

    
else:
    print("No lagrange multiplier needed")
    fes = FESpace([fescurl,fesp,fescurl, N], dgjumps = True)
    (u, p, B, rho), (v, q, C, drho) = fes.TnT()

    gf = GridFunction(fes)
    gfu, gfp, gfB, gfrho = gf.components

    gf_old = GridFunction(fes)
    gfu_old, gfp_old, gfB_old, gfrho_old = gf_old.components

    gf_star = GridFunction(fes)
    gfu_star, gfp_star, gfB_star, gfrho_star = gf_star.components

    gf_ex = GridFunction(fes)
    gfu_ex, gfp_ex, gfB_ex, gfrho_ex = gf_ex.components


p_exp = 20
L2lo = L2(mesh, order = 0)
gfB_Linf = GridFunction(L2lo)
gfu_Linf = GridFunction(L2lo)

# set initial condition
fesup = fescurl*fesp*N

n = specialcf.normal(2)
h = specialcf.mesh_size

(uu, pp, rr), (vv, qq, drr) = fesup.TnT()

stokes = BilinearForm(fesup)
stokes += mu*curl(uu)*curl(vv)*dx + grad(pp)*vv*dx + uu*grad(qq)*dx 
stokes += mu*curl(uu)*cross2d(vv, n)*ds(skeleton = True) + mu*curl(vv)*cross2d(uu,n)*ds(skeleton = True) + mu*gammaNitsche/h * uu.Trace()*vv.Trace()*ds
stokes += pp*drr*dx + rr*qq*dx
stokes.Assemble()

bf_lift = BilinearForm(fesup)
bf_lift += uu*vv*dx + grad(pp)*vv*dx + uu*grad(qq)*dx 
bf_lift += pp*drr*dx + rr*qq*dx

bf_lift.Assemble()
gf_lift = GridFunction(fesup)

rhs_lift = LinearForm(fesup)
if periodic_x == False or periodic_y == False:
    rhs_lift += InnerProduct(Bbnd, n)*qq*ds

rhsup = LinearForm(fesup)
if periodic_x == False or periodic_y == False:
    rhsup += mu*gammaNitsche/h * cross2d(ubnd, n)*cross2d(vv, n)*ds(skeleton = True)
    rhsup += mu*curl(vv)*cross2d(ubnd,n)*ds(skeleton = True) 
rhsup.Assemble()


bfup = BilinearForm(fesup)
bfup += uu*vv*dx + grad(pp)*vv*dx + uu*grad(qq)*dx 
bfup += pp*drr*dx + rr*qq*dx 
bfup.Assemble()

bfcurlcurl = BilinearForm(fesup)
bfcurlcurl += curl(uu)*curl(vv)*dx + grad(pp)*vv*dx  + uu*grad(qq)*dx 
bfcurlcurl += pp*drr*dx +rr*qq*dx 
bfcurlcurl.Assemble()

rhsu = LinearForm(fesup)
rhsu += u0*vv*dx 
rhsu += InnerProduct(u0, n)*qq*ds(skeleton = True)
rhsu.Assemble()

rhsB = LinearForm(fesup)
if ictype == 7:
    rhsB += CF((0,0))*vv*dx
else:
    rhsB += B0*vv*dx 
rhsB += InnerProduct(B0, n)*qq*ds(skeleton = True)
rhsB.Assemble()

gf_ic = GridFunction(fesup)
gfu_ic, gfp_ic, gfrho_ic = gf_ic.components

if ictype == 15:
    gf_ic.vec.data = stokes.mat.Inverse(freedofs = fesup.FreeDofs())*rhsup.vec
else:
    gf_ic.vec.data = bfup.mat.Inverse(freedofs = fesup.FreeDofs())*rhsu.vec
gfu.vec.data = gfu_ic.vec

if ictype == 7:
    gf_ic.vec.data = bfcurlcurl.mat.Inverse(freedofs = fesup.FreeDofs()) * rhsB.vec
else:
    gf_ic.vec.data = bfup.mat.Inverse(freedofs = fesup.FreeDofs()) * rhsB.vec

gfB.vec.data = gfu_ic.vec
# for ictype 7
gfB0 = GridFunction(fescurl)
gfu0 = GridFunction(fescurl)

gfB0.vec.data = gfB.vec
gfu0.vec.data = gfu.vec

# Initial error 
print("Err L2 for u is " + str(sqrt(Integrate(Norm(u0 - gfu)**2, mesh))))
print("Err L2 for B is " + str(sqrt(Integrate(Norm(B0 - gfB)**2, mesh))))
print("Curl B = " + str( sqrt(Integrate(Norm(curl(gfB))**2, mesh))))



diff = curl(u)*curl(v)*dx 
if periodic_x == False or periodic_y == False:
    diff += curl(u)*cross2d(v, n)*ds(skeleton = True) + curl(v)*cross2d(u,n)*ds(skeleton = True) + gammaNitsche/h * cross2d(u,n)*cross2d(v,n)*ds(skeleton = True)


normB = my_max(C_S, my_max(gfB_Linf, gfB_Linf.Other()))
normu = my_max(C_S, my_max(gfu_Linf, gfu_Linf.Other()))


stab =  gammaCIPu_strong/h* normu*InnerProduct(u- u.Other(), v- v.Other())*dx(skeleton = True)
stab += gammaCIPu* normu*InnerProduct(u- u.Other(), v- v.Other())*dx(skeleton = True)
if periodic_x == False or periodic_y == False:
    stab += gammaCIPu *  my_max( my_max(C_S, gfu_Linf), my_max(C_S, gfB_Linf)) * InnerProduct(u,n)*InnerProduct(v,n)*ds(skeleton = True) # 
stab += gammaCIPgradu * normB* h**2 *InnerProduct(grad(u) - grad(u.Other()), grad(v) - grad(v.Other()))*dx(skeleton  =True)
stab += gammaCIPcurlB * (normu + normB)*h**2 *InnerProduct( curl(B) - curl(B.Other()), curl(C) - curl(C.Other()))*dx(skeleton = True)
stab += gammaCIPB/h * normu*InnerProduct(B- B.Other(),C- C.Other())*dx(skeleton = True)

bf_stab = BilinearForm( fes , nonassemble = True)
bf_stab += stab

bf_curl = BilinearForm(fes, nonassemble = True)
bf_curl += mu*diff #+ eta*curl(B)*curl(C)*dx

gff = GridFunction(fescurl)

bf = BilinearForm(fes, nonassemble = True)
bf += 2./dt*u*v*dx 
bf += cross2d(u,v)*curl(u)*dx(bonus_intorder = 3)
bf += mu*diff
if periodic_x == False or periodic_y == False:
    bf += -  mu*curl(v)*cross2d(ubnd,n)*ds(skeleton = True) - mu*gammaNitsche/h * cross2d(ubnd,n)*cross2d(v, n)*ds(skeleton = True)
bf += grad(p)*v*dx
bf += u*grad(q)*dx
if periodic_x == False or periodic_y == False:
    bf +=  -InnerProduct(ubnd,n)*q*ds(skeleton = True)
bf += - curl(B)*cross2d(B, v)*dx(bonus_intorder = 3)
bf += 2./dt*B*C*dx 
bf += cross2d(B, u)*curl(C)*dx(bonus_intorder = 3) 
bf += eta*curl(B)*curl(C)*dx
bf += -2./dt*gfB_old*C*dx - 2./dt*gfu_old*v*dx
bf += p*drho*dx + rho*q*dx 
bf += -gff*v*dx
bf += -g*C*dx

# bc 
if periodic_x == False or periodic_y == False:
    bf += Ebnd*cross2d(C, n)*ds(skeleton = True)
    # # lift for nonhomogeneous BC for B
    # bf += cross2d(gfB_lift, u)*curl(C)*dx(bonus_intorder = 3)
    # bf += 2/dt*InnerProduct(gfB_lift, C)*dx + eta*curl(gfB_lift)*curl(C)*dx 

# stabilizations 
bf += stab
if periodic_x == False or periodic_y == False:
    print("Rhs stab")
    stab += -gammaCIPu * my_max( my_max(C_S, gfu_Linf), my_max(C_S, gfB_Linf)) * InnerProduct(ubnd,n)*InnerProduct(v, n)*ds(skeleton = True) # 

# if we need a Lagrange multiplier also for B
if lag_mult:
    bf += grad(phi)*C*dx + B*grad(psi)*dx
    bf += zeta*psi*dx + phi*dzeta*dx
    if periodic_x == False or periodic_y == False:
        bf += - Bbnd*n*psi*ds


bf_lin = BilinearForm(fes)
bf_lin += 2./dt*u*v*dx 
bf_lin += cross2d(gfu_star,v)*curl(u)*dx(bonus_intorder = 3)
bf_lin += cross2d(u,v)*curl(gfu_star)*dx(bonus_intorder = 3)
bf_lin += mu*diff
bf_lin += grad(p)*v*dx
bf_lin += u*grad(q)*dx
bf_lin += - curl(gfB_star)*cross2d(B, v)*dx(bonus_intorder = 3)
bf_lin += - curl(B)*cross2d(gfB_star, v)*dx(bonus_intorder = 3)
bf_lin += 2./dt* B*C*dx
bf_lin += cross2d(gfB_star, u)*curl(C)*dx(bonus_intorder = 3) 
bf_lin += cross2d(B, gfu_star)*curl(C)*dx(bonus_intorder = 3) 
bf_lin += eta*curl(B)*curl(C)*dx
bf_lin += p*drho*dx + rho*q*dx 

# stabilization
bf_lin += stab

if lag_mult:
    bf_lin += grad(phi)*C*dx + B*grad(psi)*dx
    bf_lin += phi*dzeta*dx + zeta*psi*dx


def SimpleNewtonSolve(tol=1e-13,maxits=25):
    res = gf_star.vec.CreateVector()
    du = gf_star.vec.CreateVector()
    bf.Apply(gf_star.vec, res)
    for it in range(maxits):
        print ("Iteration {:3}  ".format(it),end="")

        bf_lin.Assemble()

        du[:] = 0
        du.data = bf_lin.mat.Inverse(fes.FreeDofs(), inverse = "pardiso") * res
        gf_star.vec.data -= du
        bf.Apply(gf_star.vec, res)

        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(res,res)))
        print ("<A u",it,", A u",it,">_{-1}^0.5 = ", stopcritval)
        if stopcritval > 100:
            return -1
        if stopcritval < tol:
            return 0
            
    return -1

variant  = ""

def writeData(gfu, gfB, gfp, title):


    fesplot = H1(mesh, order = order)
    
    gfuxplot = GridFunction(fesplot)
    gfuyplot = GridFunction(fesplot)
    gfBx = GridFunction(fesplot)
    gfBy = GridFunction(fesplot)
    gfBmod = GridFunction(fesplot)
    gfumod = GridFunction(fesplot)

    gfuxplot.Set(gfu[0])
    gfuyplot.Set(gfu[1])

    gfBx.Set(gfB[0])
    gfBy.Set(gfB[1])

    gfBmod.Set(Norm(gfB))
    gfumod.Set(Norm(gfu))

    vtk = VTKOutput(ma=mesh,
                coefs=[gfu, gfuxplot, gfuyplot, gfp, gfB, gfBx, gfBy, gfBmod, gfumod],
                names = ["uvec", "u","v", "p", "B", "Bx", "By", "|B|", "|u|"],
                subdivision = 3,
                filename=icname + "_" + str(NMAX) + "_" + method_name +title)
    # Exporting the results:
    vtk.Do()



writeData(gfu, gfB, gfp, "start")

t = 0
i = 0

from ngsolve.webgui import *

E = []
CH = [] # cross helicity 
tvec = [] # time vector

E.append(0.5*Integrate(gfu**2 + gfB**2, mesh))
CH.append(Integrate(gfu*gfB, mesh))
tvec.append(0)



vec_aux = gf_star.vec.CreateVector()
err_vec = gf_star.vec.CreateVector()
errL2 = []
errCurl = []
errStab = []

while t < tend - 1e-8:

    gf_old.vec.data = gf.vec

    LinftyB =  Integrate(Norm(gfB_old) ** p_exp, mesh, element_wise=True)
    gfB_Linf.vec.FV().NumPy()[:] = np.array(LinftyB)**(1./p_exp)


    Linftyu =  Integrate(Norm(gfu_old) ** p_exp, mesh, element_wise=True)
    gfu_Linf.vec.FV().NumPy()[:] = np.array(Linftyu)**(1./p_exp)

    gf_star.vec.data = gf_old.vec # initialize for Newton iteration

    time.Set(t + 0.5*dt) # for the midpoint rule
    gff.Set(f, dual = True)

    # compute lifting 
    # if periodic_y and periodic_x:
    #     gfB_lift.Set(CF((0,0))) 
    # else:
    #     rhs_lift.Assemble()
    #     gf_lift.vec.data = bf_lift.mat.Inverse(freedofs = fesup.FreeDofs())* rhs_lift.vec
    #     gfB_lift.vec.data = gf_lift.components[0].vec

    SimpleNewtonSolve(tol = 1e-10)

    # add lift
    # gfB_star.vec.data += gfB_lift.vec
    
    # recover gf^{n+1} = 2*gf^{n+ 1/2} - gf^n
    gf.vec.data = 2*gf_star.vec - gf_old.vec
    # but not for the pressure
    gfp.vec.data = gfp_star.vec
    #

    t += dt
    E.append(0.5*Integrate(gfu**2 + gfB**2, mesh))
    CH.append(Integrate(gfu*gfB, mesh))
    tvec.append(t)

    # check
    print("Curl B = " + str( sqrt(Integrate(Norm(curl(gfB))**2, mesh))))

    time.Set(t) # to compute the error
    if ictype == 1 or ictype == 6 or ictype == 7:
        # compute errors
        errL2.append( Integrate(Norm(gfu-ufun)**2, mesh) + Integrate(Norm(gfB - Bfun)**2, mesh) )
        vec_aux[:] = 0
        if ictype == 7:
            gfu_ex.vec.data = gfu0.vec
            gfB_ex.vec.data = gfB0.vec
        else:
            gfu_ex.Set(ufun, dual = True)
            gfB_ex.Set(Bfun)
        err_vec.data = gf.vec - gf_ex.vec
        bf_curl.Apply(err_vec, vec_aux)
        errCurl.append( InnerProduct(err_vec, vec_aux) + eta*Integrate((curlBfun - curl(gfB))**2, mesh ) )
        if ictype == 7:
            print(sqrt(errCurl))
        vec_aux[:] = 0
        bf_stab.Apply(err_vec, vec_aux)
        errStab.append( InnerProduct(err_vec, vec_aux))


    i = i +1

    if i%nplot == 0:
        writeData(gfu, gfB, gfp, str(t))
        print(t)

time.Set(tend)

if ictype == 1 or ictype == 6 or ictype == 7:

    errL2 = np.array(errL2)
    errCurl = np.array(errCurl)
    errStab = np.array(errStab)
    #
    int0t_errL2 = dt*0.5*np.sum(errL2[:-1] + errL2[1:])
    int0t_errH1 = dt*0.5*np.sum(errCurl[:-1] + errCurl[1:])
    int0t_errstab = dt*0.5*np.sum(errStab[:-1] + errStab[1:])
    #
    errLinftyL2 = np.max(np.sqrt(errL2))
    errL2H1 = sqrt( int0t_errL2 + int0t_errH1 )
    err_tot = errLinftyL2 + sqrt( int0t_errL2 + int0t_errH1 + int0t_errstab)
    print(errLinftyL2, errL2H1, err_tot)

    # ---------------------------------------------------------
    # SAVE RESULTS TO FILE (APPEND MODE)
    # ---------------------------------------------------------
    import os
    
    # You can change this filename if you want different batches
    output_file = "results" + str(ictype) + ".txt"
    
    # Check if file exists so we know whether to write the header
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, "a") as f:
        # 1. Write Header (only if file is new)
        if not file_exists:
            f.write("ictype,NMAX,order,dt,method_type,mu,eta,tend,errLinftyL2,errL2H1,err_tot\n")
        
        # 2. Write Data (Append to bottom)
        # Using .10e for high precision scientific notation
        f.write(f"{ictype},{NMAX},{order},{dt},{method_type},{mu},{eta},{tend},"
                f"{errLinftyL2:.10e},{errL2H1:.10e},{err_tot:.10e}\n")

    print(f"Results appended to {output_file}")

tvec = np.array(tvec)
E = np.array(E)
CH = np.array(CH)

dat = np.array([tvec, E, CH])
dat = dat.T
np.savetxt(icname + "_" + str(NMAX) + "_" + method_name + "_cons.csv", dat, delimiter=",")
writeData(gfu, gfB, gfp, "_tend")