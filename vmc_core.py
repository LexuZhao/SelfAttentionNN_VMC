# {
#  "cells": [
#   {
#    "cell_type": "code",
#    "execution_count": 8,
#    "id": "4dcfcf90",
#    "metadata": {},
#    "outputs": [
#     {
#      "ename": "RecursionError",
#      "evalue": "maximum recursion depth exceeded while calling a Python object",
#      "output_type": "error",
#      "traceback": [
#       "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
#       "\u001b[0;31mRecursionError\u001b[0m                            Traceback (most recent call last)",
#       "Cell \u001b[0;32mIn[8], line 90\u001b[0m\n\u001b[1;32m     87\u001b[0m cfgs  \u001b[38;5;241m=\u001b[39m sample_configs(alpha)\n\u001b[1;32m     89\u001b[0m \u001b[38;5;66;03m# local energies and O = ∂lnΨ/∂α = -Σ r²\u001b[39;00m\n\u001b[0;32m---> 90\u001b[0m E_loc \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39marray(\u001b[43m[\u001b[49m\u001b[43mlocal_energy\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43malpha\u001b[49m\u001b[43m)\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43;01mfor\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mR\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;129;43;01min\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mcfgs\u001b[49m\u001b[43m]\u001b[49m)\n\u001b[1;32m     91\u001b[0m O     \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m-\u001b[39mnp\u001b[38;5;241m.\u001b[39marray([np\u001b[38;5;241m.\u001b[39msum(R\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m2\u001b[39m) \u001b[38;5;28;01mfor\u001b[39;00m R \u001b[38;5;129;01min\u001b[39;00m cfgs])\n\u001b[1;32m     93\u001b[0m E_mean \u001b[38;5;241m=\u001b[39m E_loc\u001b[38;5;241m.\u001b[39mmean()\n",
#       "Cell \u001b[0;32mIn[8], line 90\u001b[0m, in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m     87\u001b[0m cfgs  \u001b[38;5;241m=\u001b[39m sample_configs(alpha)\n\u001b[1;32m     89\u001b[0m \u001b[38;5;66;03m# local energies and O = ∂lnΨ/∂α = -Σ r²\u001b[39;00m\n\u001b[0;32m---> 90\u001b[0m E_loc \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39marray([\u001b[43mlocal_energy\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43malpha\u001b[49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mfor\u001b[39;00m R \u001b[38;5;129;01min\u001b[39;00m cfgs])\n\u001b[1;32m     91\u001b[0m O     \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m-\u001b[39mnp\u001b[38;5;241m.\u001b[39marray([np\u001b[38;5;241m.\u001b[39msum(R\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m2\u001b[39m) \u001b[38;5;28;01mfor\u001b[39;00m R \u001b[38;5;129;01min\u001b[39;00m cfgs])\n\u001b[1;32m     93\u001b[0m E_mean \u001b[38;5;241m=\u001b[39m E_loc\u001b[38;5;241m.\u001b[39mmean()\n",
#       "Cell \u001b[0;32mIn[8], line 45\u001b[0m, in \u001b[0;36mlocal_energy\u001b[0;34m(R, alpha)\u001b[0m\n\u001b[1;32m     44\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mlocal_energy\u001b[39m(R, alpha):\n\u001b[0;32m---> 45\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m local_kinetic(R, alpha) \u001b[38;5;241m+\u001b[39m \u001b[43menergy_moire\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:106\u001b[0m, in \u001b[0;36menergy_moire\u001b[0;34m(R)\u001b[0m\n\u001b[1;32m    104\u001b[0m ext \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39msum(moire_potential(R))\n\u001b[1;32m    105\u001b[0m ee  \u001b[38;5;241m=\u001b[39m coulomb_ewald_2D(R)\n\u001b[0;32m--> 106\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_total_moire  =\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[43menergy_moire\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    107\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mwhere:\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m    108\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_ee =\u001b[39m\u001b[38;5;124m\"\u001b[39m, coulomb_ewald_2D(R), \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:106\u001b[0m, in \u001b[0;36menergy_moire\u001b[0;34m(R)\u001b[0m\n\u001b[1;32m    104\u001b[0m ext \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39msum(moire_potential(R))\n\u001b[1;32m    105\u001b[0m ee  \u001b[38;5;241m=\u001b[39m coulomb_ewald_2D(R)\n\u001b[0;32m--> 106\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_total_moire  =\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[43menergy_moire\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    107\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mwhere:\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m    108\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_ee =\u001b[39m\u001b[38;5;124m\"\u001b[39m, coulomb_ewald_2D(R), \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
#       "    \u001b[0;31m[... skipping similar frames: energy_moire at line 106 (2963 times)]\u001b[0m\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:106\u001b[0m, in \u001b[0;36menergy_moire\u001b[0;34m(R)\u001b[0m\n\u001b[1;32m    104\u001b[0m ext \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39msum(moire_potential(R))\n\u001b[1;32m    105\u001b[0m ee  \u001b[38;5;241m=\u001b[39m coulomb_ewald_2D(R)\n\u001b[0;32m--> 106\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_total_moire  =\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[43menergy_moire\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    107\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mwhere:\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m    108\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_ee =\u001b[39m\u001b[38;5;124m\"\u001b[39m, coulomb_ewald_2D(R), \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:105\u001b[0m, in \u001b[0;36menergy_moire\u001b[0;34m(R)\u001b[0m\n\u001b[1;32m    103\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21menergy_moire\u001b[39m(R):\n\u001b[1;32m    104\u001b[0m     ext \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39msum(moire_potential(R))\n\u001b[0;32m--> 105\u001b[0m     ee  \u001b[38;5;241m=\u001b[39m \u001b[43mcoulomb_ewald_2D\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    106\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_total_moire  =\u001b[39m\u001b[38;5;124m\"\u001b[39m, energy_moire(R), \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    107\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mwhere:\u001b[39m\u001b[38;5;124m'\u001b[39m)\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:100\u001b[0m, in \u001b[0;36mcoulomb_ewald_2D\u001b[0;34m(R)\u001b[0m\n\u001b[1;32m     95\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m     96\u001b[0m \u001b[38;5;124;03mFull 2D Ewald Coulomb energy for N unit charges in a square PBC box:\u001b[39;00m\n\u001b[1;32m     97\u001b[0m \u001b[38;5;124;03m  E_total = (e² / (4πϵ₀ ε_r)) · ( E_real + E_recip + E_self );  R : (N,2) array of positions [nm]; returns energy [meV].\u001b[39;00m\n\u001b[1;32m     98\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m     99\u001b[0m N \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlen\u001b[39m(R)\n\u001b[0;32m--> 100\u001b[0m E \u001b[38;5;241m=\u001b[39m pairwise_real_space(R) \u001b[38;5;241m+\u001b[39m \u001b[43mreciprocal_space\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m \u001b[38;5;241m+\u001b[39m self_energy(N)\n\u001b[1;32m    101\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m E \u001b[38;5;241m*\u001b[39m e2_4pieps0 \u001b[38;5;241m/\u001b[39m eps_r\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:82\u001b[0m, in \u001b[0;36mreciprocal_space\u001b[0;34m(R, alpha, k_lim, L)\u001b[0m\n\u001b[1;32m     80\u001b[0m         k2 \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39mdot(kvec, kvec)         \u001b[38;5;66;03m# q² = |q_vec|² = q_x² + q_y²\u001b[39;00m\n\u001b[1;32m     81\u001b[0m         damp \u001b[38;5;241m=\u001b[39m exp(\u001b[38;5;241m-\u001b[39mk2 \u001b[38;5;241m/\u001b[39m (\u001b[38;5;241m4.0\u001b[39m \u001b[38;5;241m*\u001b[39m alpha)) \u001b[38;5;66;03m# factor e^{–q²/(4α)}\u001b[39;00m\n\u001b[0;32m---> 82\u001b[0m         Sq2 \u001b[38;5;241m=\u001b[39m \u001b[43mstructure_factor\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mkvec\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     83\u001b[0m         E_k \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m (damp \u001b[38;5;241m*\u001b[39m np\u001b[38;5;241m.\u001b[39mabs(Sq2) \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39m \u001b[38;5;241m2\u001b[39m) \u001b[38;5;241m/\u001b[39m k2 \u001b[38;5;66;03m# e^{-q²/(4α)}/q² · |S(q)|²\u001b[39;00m\n\u001b[1;32m     84\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m (pi \u001b[38;5;241m/\u001b[39m area) \u001b[38;5;241m*\u001b[39m E_k\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:59\u001b[0m, in \u001b[0;36mstructure_factor\u001b[0;34m(R, kvec)\u001b[0m\n\u001b[1;32m     55\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m     56\u001b[0m \u001b[38;5;124;03mFourier component: S(k) = ∑_j e^{-i k · r_j} Computed via cos/sin: ∑ cos(k·r) + i ∑ sin(k·r)\u001b[39;00m\n\u001b[1;32m     57\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m     58\u001b[0m phase \u001b[38;5;241m=\u001b[39m R \u001b[38;5;241m@\u001b[39m kvec\n\u001b[0;32m---> 59\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mnp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43msum\u001b[49m\u001b[43m(\u001b[49m\u001b[43mnp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcos\u001b[49m\u001b[43m(\u001b[49m\u001b[43mphase\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m \u001b[38;5;241m+\u001b[39m \u001b[38;5;241m1\u001b[39mj \u001b[38;5;241m*\u001b[39m np\u001b[38;5;241m.\u001b[39msum(np\u001b[38;5;241m.\u001b[39msin(phase))\n",
#       "File \u001b[0;32m/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/numpy/core/fromnumeric.py:2313\u001b[0m, in \u001b[0;36msum\u001b[0;34m(a, axis, dtype, out, keepdims, initial, where)\u001b[0m\n\u001b[1;32m   2310\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m out\n\u001b[1;32m   2311\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m res\n\u001b[0;32m-> 2313\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43m_wrapreduction\u001b[49m\u001b[43m(\u001b[49m\u001b[43ma\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43madd\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43msum\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43maxis\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdtype\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mout\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mkeepdims\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mkeepdims\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   2314\u001b[0m \u001b[43m                      \u001b[49m\u001b[43minitial\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43minitial\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mwhere\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mwhere\u001b[49m\u001b[43m)\u001b[49m\n",
#       "File \u001b[0;32m/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/numpy/core/fromnumeric.py:72\u001b[0m, in \u001b[0;36m_wrapreduction\u001b[0;34m(obj, ufunc, method, axis, dtype, out, **kwargs)\u001b[0m\n\u001b[1;32m     71\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_wrapreduction\u001b[39m(obj, ufunc, method, axis, dtype, out, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[0;32m---> 72\u001b[0m     passkwargs \u001b[38;5;241m=\u001b[39m {k: v \u001b[38;5;28;01mfor\u001b[39;00m k, v \u001b[38;5;129;01min\u001b[39;00m kwargs\u001b[38;5;241m.\u001b[39mitems()\n\u001b[1;32m     73\u001b[0m                   \u001b[38;5;28;01mif\u001b[39;00m v \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m np\u001b[38;5;241m.\u001b[39m_NoValue}\n\u001b[1;32m     75\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mtype\u001b[39m(obj) \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m mu\u001b[38;5;241m.\u001b[39mndarray:\n\u001b[1;32m     76\u001b[0m         \u001b[38;5;28;01mtry\u001b[39;00m:\n",
#       "\u001b[0;31mRecursionError\u001b[0m: maximum recursion depth exceeded while calling a Python object"
#      ]
#     }
#    ],
#    "source": [
#     "\"\"\"\n",
#     "Tiny but *stable* VMC loop for the Gaussian trial wave‑function\n",
#     "Key fixes compared with the first draft\n",
#     "1. Optimise **log‑alpha** so α = exp(log_alpha) is always positive.\n",
#     "2. Use a realistic kinetic prefactor  ħ²/(2m*) ≈ 3500 meV·nm².\n",
#     "3. Smaller learning‑rate 1e‑4  and **gradient clipping** to ±100.\n",
#     "4. Slightly smaller proposal step δ = 0.3 nm for higher acceptance.\n",
#     "\"\"\"\n",
#     "import numpy as np\n",
#     "import importlib\n",
#     "import matplotlib.pyplot as plt\n",
#     "from math import exp, log\n",
#     "# rmb to Import the physics Hamiltonian  energy_moire(R)\n",
#     "\n",
#     "# 1. import energy_moire\n",
#     "try:\n",
#     "    moire_model = importlib.import_module(\"moire_model\")\n",
#     "    energy_moire = moire_model.energy_moire\n",
#     "    L = moire_model.a_m\n",
#     "except ModuleNotFoundError:\n",
#     "    print(\"WARNING: moire_model.py not found; using potential=0 for demo.\")\n",
#     "#     L = 8.5\n",
#     "#     def energy_moire(R):\n",
#     "#         return 0.0\n",
#     "#     moire_model = types.SimpleNamespace(a_m=L)\n",
#     "\n",
#     "# L = a_m\n",
#     "\n",
#     "\n",
#     "# -------------------------------------------------------------------------\n",
#     "# Wave‑function\n",
#     "def log_psi(R, alpha):          # Ψ(R) = exp(log(Ψ(R))) with log(Ψ(R)) = -α * sum_i(|r_i|²)\n",
#     "    return -alpha * np.sum(R**2)            # ln Ψ  (Ψ>0 so sign OK)\n",
#     "\n",
#     "def psi_sq(R, alpha):\n",
#     "    return np.exp(2.0 * log_psi(R, alpha))    # probability density: |Ψ(R)|² = Ψ(R)² = exp[2 * log(Ψ(R))]\n",
#     "\n",
#     "def local_kinetic(R, alpha, hbar2_over_2m=3500.0):\n",
#     "    \"\"\" Analytic kinetic energy of the Gaussian in *meV*.\"\"\"\n",
#     "    N = len(R)\n",
#     "    r2 = np.sum(R**2)                         # r² = sum_{i=1 to N} (r_i,x² + r_i,y²) = sum_{i=1 to N} |r_i|²\n",
#     "    return hbar2_over_2m * 4.0 * (alpha * N - alpha**2 * r2)      # T_loc(R) = -(ħ²/2m)*(∇²Ψ(R)/Ψ(R)) = (ħ²/2m)*4*(α*N - α²*sum_i(r_i²))\n",
#     "\n",
#     "def local_energy(R, alpha):\n",
#     "    return local_kinetic(R, alpha) + energy_moire(R)\n",
#     "\n",
#     "# -------------------------------------------------------------------------\n",
#     "# Metropolis–Hastings sampler\n",
#     "# We’re using Metropolis–Hastings to draw random electron configurations according to our trial wavefunction’s probability density,\n",
#     "# so that we can estimate energies and their derivatives by simple averages over those configurations.\n",
#     "def metro_step(R, alpha, delta=0.3):\n",
#     "    \"\"\"\n",
#     "    One sweep through all electrons; returns updated configuration.\n",
#     "    \"\"\"\n",
#     "    N = len(R)\n",
#     "    for i in range(N):\n",
#     "        old = R[i].copy()\n",
#     "        new = (old + np.random.uniform(-delta, delta, 2)) % L\n",
#     "        d_logprob = 2.0 * (-alpha * (np.sum(new**2) - np.sum(old**2)))\n",
#     "        if np.log(np.random.rand()) < d_logprob:\n",
#     "            R[i] = new\n",
#     "    return R\n",
#     "\n",
#     "def sample_configs(alpha, N_e=6, n_steps=800, burn=400, delta=0.3):\n",
#     "    R = np.random.rand(N_e, 2) * L  # random start\n",
#     "    cfgs = []\n",
#     "    for step in range(n_steps):\n",
#     "        R = metro_step(R, alpha, delta)\n",
#     "        if step >= burn:\n",
#     "            cfgs.append(R.copy())\n",
#     "    return np.array(cfgs)\n",
#     "\n",
#     "# -------------------------------------------------------------------------\n",
#     "# VMC optimisation loop  (log‑alpha parameter)\n",
#     "np.random.seed(0)\n",
#     "\n",
#     "log_alpha = np.log(0.05)         # start at α=0.05 nm⁻²\n",
#     "lr        = 1e-4                 # learning‑rate # modified (check)\n",
#     "clip      = 100.0                # gradient clip # modified (check)\n",
#     "n_outer   = 30                   # optimisation iterations\n",
#     "\n",
#     "E_curve = []\n",
#     "alpha_curve = []\n",
#     "\n",
#     "for it in range(n_outer):\n",
#     "    alpha = np.exp(log_alpha) # modified (check)\n",
#     "    cfgs  = sample_configs(alpha)\n",
#     "\n",
#     "    # local energies and O = ∂lnΨ/∂α = -Σ r²\n",
#     "    E_loc = np.array([local_energy(R, alpha) for R in cfgs])\n",
#     "    O     = -np.array([np.sum(R**2) for R in cfgs])\n",
#     "\n",
#     "    E_mean = E_loc.mean()\n",
#     "    grad   = 2.0 * ((E_loc * O).mean() - E_mean * O.mean())   # dE/dα\n",
#     "    grad   = np.clip(grad, -clip, clip)\n",
#     "\n",
#     "    log_alpha -= lr * grad      # gradient descent in log‑space\n",
#     "\n",
#     "    E_curve.append(E_mean)\n",
#     "    alpha_curve.append(alpha)\n",
#     "    print(f\"{it:02d}  ⟨E⟩ = {E_mean:10.5f} meV   α = {alpha:7.4f} nm⁻²   grad = {grad:+7.3f}\")\n",
#     "\n",
#     "# -------------------------------------------------------------------------\n",
#     "# 5)  Plot learning curves\n",
#     "fig, ax = plt.subplots(1, 2, figsize=(8,3))\n",
#     "ax[0].plot(E_curve, marker=\"o\")\n",
#     "ax[0].set_xlabel(\"iteration\")\n",
#     "ax[0].set_ylabel(\"energy ⟨E⟩ (meV)\")\n",
#     "ax[0].set_title(\"VMC energy\")\n",
#     "ax[1].plot(alpha_curve, marker=\"s\", color=\"tab:orange\")\n",
#     "ax[1].set_xlabel(\"iteration\")\n",
#     "ax[1].set_ylabel(\"α (nm⁻²)\")\n",
#     "ax[1].set_title(\"α evolution\")\n",
#     "for a in ax: a.grid(alpha=0.3)\n",
#     "fig.tight_layout()\n",
#     "plt.show()"
#    ]
#   },
#   {
#    "cell_type": "code",
#    "execution_count": 9,
#    "id": "22e0721c",
#    "metadata": {},
#    "outputs": [
#     {
#      "ename": "RecursionError",
#      "evalue": "maximum recursion depth exceeded while calling a Python object",
#      "output_type": "error",
#      "traceback": [
#       "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
#       "\u001b[0;31mRecursionError\u001b[0m                            Traceback (most recent call last)",
#       "Cell \u001b[0;32mIn[9], line 61\u001b[0m\n\u001b[1;32m     59\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m it \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mrange\u001b[39m(n_iter):\n\u001b[1;32m     60\u001b[0m     cfgs\u001b[38;5;241m=\u001b[39mgenerate_samples(alpha)\n\u001b[0;32m---> 61\u001b[0m     E_loc\u001b[38;5;241m=\u001b[39mnp\u001b[38;5;241m.\u001b[39marray(\u001b[43m[\u001b[49m\u001b[43mlocal_energy\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43malpha\u001b[49m\u001b[43m)\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43;01mfor\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mR\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;129;43;01min\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43mcfgs\u001b[49m\u001b[43m]\u001b[49m)\n\u001b[1;32m     62\u001b[0m     O\u001b[38;5;241m=\u001b[39mnp\u001b[38;5;241m.\u001b[39marray([\u001b[38;5;241m-\u001b[39mnp\u001b[38;5;241m.\u001b[39msum(R\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m2\u001b[39m) \u001b[38;5;28;01mfor\u001b[39;00m R \u001b[38;5;129;01min\u001b[39;00m cfgs])\n\u001b[1;32m     63\u001b[0m     E_mean\u001b[38;5;241m=\u001b[39mE_loc\u001b[38;5;241m.\u001b[39mmean()\n",
#       "Cell \u001b[0;32mIn[9], line 61\u001b[0m, in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m     59\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m it \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mrange\u001b[39m(n_iter):\n\u001b[1;32m     60\u001b[0m     cfgs\u001b[38;5;241m=\u001b[39mgenerate_samples(alpha)\n\u001b[0;32m---> 61\u001b[0m     E_loc\u001b[38;5;241m=\u001b[39mnp\u001b[38;5;241m.\u001b[39marray([\u001b[43mlocal_energy\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43malpha\u001b[49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mfor\u001b[39;00m R \u001b[38;5;129;01min\u001b[39;00m cfgs])\n\u001b[1;32m     62\u001b[0m     O\u001b[38;5;241m=\u001b[39mnp\u001b[38;5;241m.\u001b[39marray([\u001b[38;5;241m-\u001b[39mnp\u001b[38;5;241m.\u001b[39msum(R\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m2\u001b[39m) \u001b[38;5;28;01mfor\u001b[39;00m R \u001b[38;5;129;01min\u001b[39;00m cfgs])\n\u001b[1;32m     63\u001b[0m     E_mean\u001b[38;5;241m=\u001b[39mE_loc\u001b[38;5;241m.\u001b[39mmean()\n",
#       "Cell \u001b[0;32mIn[9], line 31\u001b[0m, in \u001b[0;36mlocal_energy\u001b[0;34m(R, alpha)\u001b[0m\n\u001b[1;32m     30\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mlocal_energy\u001b[39m(R, alpha):\n\u001b[0;32m---> 31\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m local_kinetic(R, alpha) \u001b[38;5;241m+\u001b[39m \u001b[43menergy_moire\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:106\u001b[0m, in \u001b[0;36menergy_moire\u001b[0;34m(R)\u001b[0m\n\u001b[1;32m    104\u001b[0m ext \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39msum(moire_potential(R))\n\u001b[1;32m    105\u001b[0m ee  \u001b[38;5;241m=\u001b[39m coulomb_ewald_2D(R)\n\u001b[0;32m--> 106\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_total_moire  =\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[43menergy_moire\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    107\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mwhere:\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m    108\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_ee =\u001b[39m\u001b[38;5;124m\"\u001b[39m, coulomb_ewald_2D(R), \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:106\u001b[0m, in \u001b[0;36menergy_moire\u001b[0;34m(R)\u001b[0m\n\u001b[1;32m    104\u001b[0m ext \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39msum(moire_potential(R))\n\u001b[1;32m    105\u001b[0m ee  \u001b[38;5;241m=\u001b[39m coulomb_ewald_2D(R)\n\u001b[0;32m--> 106\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_total_moire  =\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[43menergy_moire\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    107\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mwhere:\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m    108\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_ee =\u001b[39m\u001b[38;5;124m\"\u001b[39m, coulomb_ewald_2D(R), \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
#       "    \u001b[0;31m[... skipping similar frames: energy_moire at line 106 (2963 times)]\u001b[0m\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:106\u001b[0m, in \u001b[0;36menergy_moire\u001b[0;34m(R)\u001b[0m\n\u001b[1;32m    104\u001b[0m ext \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39msum(moire_potential(R))\n\u001b[1;32m    105\u001b[0m ee  \u001b[38;5;241m=\u001b[39m coulomb_ewald_2D(R)\n\u001b[0;32m--> 106\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_total_moire  =\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[43menergy_moire\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    107\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mwhere:\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m    108\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_ee =\u001b[39m\u001b[38;5;124m\"\u001b[39m, coulomb_ewald_2D(R), \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:105\u001b[0m, in \u001b[0;36menergy_moire\u001b[0;34m(R)\u001b[0m\n\u001b[1;32m    103\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21menergy_moire\u001b[39m(R):\n\u001b[1;32m    104\u001b[0m     ext \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39msum(moire_potential(R))\n\u001b[0;32m--> 105\u001b[0m     ee  \u001b[38;5;241m=\u001b[39m \u001b[43mcoulomb_ewald_2D\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    106\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mE_total_moire  =\u001b[39m\u001b[38;5;124m\"\u001b[39m, energy_moire(R), \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmeV\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    107\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mwhere:\u001b[39m\u001b[38;5;124m'\u001b[39m)\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:100\u001b[0m, in \u001b[0;36mcoulomb_ewald_2D\u001b[0;34m(R)\u001b[0m\n\u001b[1;32m     95\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m     96\u001b[0m \u001b[38;5;124;03mFull 2D Ewald Coulomb energy for N unit charges in a square PBC box:\u001b[39;00m\n\u001b[1;32m     97\u001b[0m \u001b[38;5;124;03m  E_total = (e² / (4πϵ₀ ε_r)) · ( E_real + E_recip + E_self );  R : (N,2) array of positions [nm]; returns energy [meV].\u001b[39;00m\n\u001b[1;32m     98\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m     99\u001b[0m N \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlen\u001b[39m(R)\n\u001b[0;32m--> 100\u001b[0m E \u001b[38;5;241m=\u001b[39m pairwise_real_space(R) \u001b[38;5;241m+\u001b[39m \u001b[43mreciprocal_space\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m)\u001b[49m \u001b[38;5;241m+\u001b[39m self_energy(N)\n\u001b[1;32m    101\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m E \u001b[38;5;241m*\u001b[39m e2_4pieps0 \u001b[38;5;241m/\u001b[39m eps_r\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:82\u001b[0m, in \u001b[0;36mreciprocal_space\u001b[0;34m(R, alpha, k_lim, L)\u001b[0m\n\u001b[1;32m     80\u001b[0m         k2 \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39mdot(kvec, kvec)         \u001b[38;5;66;03m# q² = |q_vec|² = q_x² + q_y²\u001b[39;00m\n\u001b[1;32m     81\u001b[0m         damp \u001b[38;5;241m=\u001b[39m exp(\u001b[38;5;241m-\u001b[39mk2 \u001b[38;5;241m/\u001b[39m (\u001b[38;5;241m4.0\u001b[39m \u001b[38;5;241m*\u001b[39m alpha)) \u001b[38;5;66;03m# factor e^{–q²/(4α)}\u001b[39;00m\n\u001b[0;32m---> 82\u001b[0m         Sq2 \u001b[38;5;241m=\u001b[39m \u001b[43mstructure_factor\u001b[49m\u001b[43m(\u001b[49m\u001b[43mR\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mkvec\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     83\u001b[0m         E_k \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m (damp \u001b[38;5;241m*\u001b[39m np\u001b[38;5;241m.\u001b[39mabs(Sq2) \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39m \u001b[38;5;241m2\u001b[39m) \u001b[38;5;241m/\u001b[39m k2 \u001b[38;5;66;03m# e^{-q²/(4α)}/q² · |S(q)|²\u001b[39;00m\n\u001b[1;32m     84\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m (pi \u001b[38;5;241m/\u001b[39m area) \u001b[38;5;241m*\u001b[39m E_k\n",
#       "File \u001b[0;32m~/Desktop/SelfAttentionNN_VMC/moire_model.py:59\u001b[0m, in \u001b[0;36mstructure_factor\u001b[0;34m(R, kvec)\u001b[0m\n\u001b[1;32m     55\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m     56\u001b[0m \u001b[38;5;124;03mFourier component: S(k) = ∑_j e^{-i k · r_j} Computed via cos/sin: ∑ cos(k·r) + i ∑ sin(k·r)\u001b[39;00m\n\u001b[1;32m     57\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m     58\u001b[0m phase \u001b[38;5;241m=\u001b[39m R \u001b[38;5;241m@\u001b[39m kvec\n\u001b[0;32m---> 59\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mnp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43msum\u001b[49m\u001b[43m(\u001b[49m\u001b[43mnp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcos\u001b[49m\u001b[43m(\u001b[49m\u001b[43mphase\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m \u001b[38;5;241m+\u001b[39m \u001b[38;5;241m1\u001b[39mj \u001b[38;5;241m*\u001b[39m np\u001b[38;5;241m.\u001b[39msum(np\u001b[38;5;241m.\u001b[39msin(phase))\n",
#       "File \u001b[0;32m/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/numpy/core/fromnumeric.py:2313\u001b[0m, in \u001b[0;36msum\u001b[0;34m(a, axis, dtype, out, keepdims, initial, where)\u001b[0m\n\u001b[1;32m   2310\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m out\n\u001b[1;32m   2311\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m res\n\u001b[0;32m-> 2313\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43m_wrapreduction\u001b[49m\u001b[43m(\u001b[49m\u001b[43ma\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43madd\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43msum\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43maxis\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdtype\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mout\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mkeepdims\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mkeepdims\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   2314\u001b[0m \u001b[43m                      \u001b[49m\u001b[43minitial\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43minitial\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mwhere\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mwhere\u001b[49m\u001b[43m)\u001b[49m\n",
#       "File \u001b[0;32m/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/numpy/core/fromnumeric.py:72\u001b[0m, in \u001b[0;36m_wrapreduction\u001b[0;34m(obj, ufunc, method, axis, dtype, out, **kwargs)\u001b[0m\n\u001b[1;32m     71\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_wrapreduction\u001b[39m(obj, ufunc, method, axis, dtype, out, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[0;32m---> 72\u001b[0m     passkwargs \u001b[38;5;241m=\u001b[39m {k: v \u001b[38;5;28;01mfor\u001b[39;00m k, v \u001b[38;5;129;01min\u001b[39;00m kwargs\u001b[38;5;241m.\u001b[39mitems()\n\u001b[1;32m     73\u001b[0m                   \u001b[38;5;28;01mif\u001b[39;00m v \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m np\u001b[38;5;241m.\u001b[39m_NoValue}\n\u001b[1;32m     75\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mtype\u001b[39m(obj) \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m mu\u001b[38;5;241m.\u001b[39mndarray:\n\u001b[1;32m     76\u001b[0m         \u001b[38;5;28;01mtry\u001b[39;00m:\n",
#       "\u001b[0;31mRecursionError\u001b[0m: maximum recursion depth exceeded while calling a Python object"
#      ]
#     }
#    ],
#    "source": [
#     "import numpy as np\n",
#     "from math import exp\n",
#     "import matplotlib.pyplot as plt\n",
#     "import types, importlib\n",
#     "\n",
#     "# 1. import energy_moire\n",
#     "try:\n",
#     "    moire_model = importlib.import_module(\"moire_model\")\n",
#     "    energy_moire = moire_model.energy_moire\n",
#     "    L = moire_model.a_m\n",
#     "except ModuleNotFoundError:\n",
#     "    print(\"WARNING: moire_model.py not found; using potential=0 for demo.\")\n",
#     "    L = 8.5\n",
#     "    def energy_moire(R):\n",
#     "        return 0.0\n",
#     "    moire_model = types.SimpleNamespace(a_m=L)\n",
#     "\n",
#     "# 2. wavefunction operations\n",
#     "def log_psi(R, alpha):\n",
#     "    return -alpha * np.sum(R**2)\n",
#     "\n",
#     "def psi_squared(R, alpha):\n",
#     "    return np.exp(2*log_psi(R, alpha))\n",
#     "\n",
#     "def local_kinetic(R, alpha, hbar2_over_2m=1.0):\n",
#     "    N = len(R)\n",
#     "    sum_r2 = np.sum(R**2)\n",
#     "    return hbar2_over_2m * 4.0 * (alpha*N - alpha**2 * sum_r2)\n",
#     "\n",
#     "def local_energy(R, alpha):\n",
#     "    return local_kinetic(R, alpha) + energy_moire(R)\n",
#     "\n",
#     "# 3. metropolis sampler\n",
#     "def metropolis_step(R, alpha, delta=0.4):\n",
#     "    N = len(R)\n",
#     "    for i in range(N):\n",
#     "        old = R[i].copy()\n",
#     "        new = (old + np.random.uniform(-delta, delta, size=2)) % L\n",
#     "        d_log_prob = 2 * (-alpha*(np.sum(new**2) - np.sum(old**2)))\n",
#     "        if np.log(np.random.rand()) < d_log_prob:\n",
#     "            R[i] = new\n",
#     "    return R\n",
#     "\n",
#     "def generate_samples(alpha, N_e=6, n_steps=500, burn=250, delta=0.4):\n",
#     "    R = np.random.rand(N_e,2)*L\n",
#     "    samples=[]\n",
#     "    for step in range(n_steps):\n",
#     "        R = metropolis_step(R, alpha, delta)\n",
#     "        if step>=burn:\n",
#     "            samples.append(R.copy())\n",
#     "    return np.array(samples)\n",
#     "\n",
#     "# 4. optimization loop\n",
#     "np.random.seed(0)\n",
#     "alpha = 0.05\n",
#     "lr=0.01\n",
#     "n_iter=20\n",
#     "E_trace=[]\n",
#     "for it in range(n_iter):\n",
#     "    cfgs=generate_samples(alpha)\n",
#     "    E_loc=np.array([local_energy(R, alpha) for R in cfgs])\n",
#     "    O=np.array([-np.sum(R**2) for R in cfgs])\n",
#     "    E_mean=E_loc.mean()\n",
#     "    grad=2*((E_loc*O).mean()-E_mean*O.mean())\n",
#     "    alpha -= lr*grad\n",
#     "    E_trace.append(E_mean)\n",
#     "    print(f\"{it:02d}: E={E_mean:8.3f} meV, alpha={alpha:.4f}, grad={grad:+.4f}\")\n",
#     "\n",
#     "# 5. plot\n",
#     "plt.figure(figsize=(5,3))\n",
#     "plt.plot(E_trace, marker=\"o\")\n",
#     "plt.xlabel(\"iteration\")\n",
#     "plt.ylabel(\"Energy (meV)\")\n",
#     "plt.title(\"VMC learning curve\")\n",
#     "plt.grid(alpha=0.3)\n",
#     "plt.tight_layout()\n",
#     "plt.show()\n"
#    ]
#   }
#  ],
#  "metadata": {
#   "kernelspec": {
#    "display_name": "Python 3",
#    "language": "python",
#    "name": "python3"
#   },
#   "language_info": {
#    "codemirror_mode": {
#     "name": "ipython",
#     "version": 3
#    },
#    "file_extension": ".py",
#    "mimetype": "text/x-python",
#    "name": "python",
#    "nbconvert_exporter": "python",
#    "pygments_lexer": "ipython3",
#    "version": "3.11.2"
#   }
#  },
#  "nbformat": 4,
#  "nbformat_minor": 5
# }

import numpy as np
# vmc_core.py -  engine that knows nothing about the ansatz
class VMC:
    """
    Generic variational-Monte-Carlo engine.
    The wavefunction object must implement:
        • log_psi(R)           -> scalar
        • grad_log_psi(R)      -> (N,2) array  (∇_i lnΨ)
        • laplacian_log_psi(R) -> scalar       (Σ_i ∇²_i lnΨ)
    and expose a mutable .params attribute that the optimiser can update.
    """

    def __init__(self, wavefunction, energy_static,
                 hbar2_over_2m=3500.0, box_length=8.5):
        self.wf   = wavefunction
        self.E0   = energy_static           # external + Coulomb  (R-only) its exactly the old "energy_moire(R)"
        self.h2m  = hbar2_over_2m
        self.L    = box_length

    # ---------- M-H sampler (doesn't know what Ψ is) ----------
    def _metro_step(self, R, delta):
        logp0 = 2.0 * self.wf.log_psi(R)    # ln |Ψ|²
        for i in range(len(R)):
            trial = R.copy()
            trial[i] = (R[i] + np.random.uniform(-delta, delta, 2)) % self.L
            logp1 = 2.0 * self.wf.log_psi(trial)
            if np.log(np.random.rand()) < (logp1 - logp0):
                R[i] = trial[i]; logp0 = logp1
        return R

    def sample(self, n_cfg=400, burn=200, delta=0.3):
        R = np.random.rand(self.wf.N_e, 2) * self.L
        cfgs=[]
        for t in range(n_cfg+burn):
            R = self._metro_step(R, delta)
            if t >= burn:
                cfgs.append(R.copy())
        return np.array(cfgs)

    # ---------- local-energy helper ----------
    def local_energy(self, R):
        grad = self.wf.grad_log_psi(R)           # (N,2)
        lap  = self.wf.laplacian_log_psi(R)      # scalar
        kinetic = -self.h2m * (lap + (grad**2).sum())
        return kinetic + self.E0(R)

    # ---------- one optimisation epoch ----------
    def step(self, n_cfg=400, lr=1e-4, clip=100.0):
        cfgs = self.sample(n_cfg=n_cfg)
        E_loc = np.array([self.local_energy(R) for R in cfgs])
        deriv = self.wf.param_derivatives(cfgs)  # shape (n_cfg, n_param)
        Obar  = deriv.mean(axis=0)
        grad  = 2.0*( (E_loc[:,None]*deriv).mean(axis=0) - E_loc.mean()*Obar )
        grad  = np.clip(grad, -clip, clip)
        self.wf.params -= lr * grad
        return E_loc.mean(), grad, self.wf.params.copy()