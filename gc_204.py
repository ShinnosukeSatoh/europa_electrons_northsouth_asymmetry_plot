""" gc_204.py """
# <実行時間>
#
# <説明>
#


# %% ライブラリのインポート
from numba import jit
# from numba.experimental import jitclass
import numpy as np
import math
import matplotlib.pyplot as plt
# from matplotlib import rc
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
import time

# from numpy.lib.npyio import savez_compressed
# from multiprocessing import Pool

# from numpy.lib.function_base import _flip_dispatcher
color = ['#0F4C81', '#FF6F61', '#645394', '#84BD00', '#F6BE00', '#F7CAC9']

# matplotlib フォント設定
"""
plt.rcParams.update({'font.sans-serif': "Arial",
                     'font.family': "sans-serif",
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     })
"""

richtext = input('rich text (y) or (n): ')
if richtext == 'y':
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage{sansmath} \sansmath \usepackage{siunitx} \sisetup{detect-all}'
    #    \usepackage{helvet}     # helvetica font
    #    \usepackage{sansmath}   # math-font matching  helvetica
    #    \sansmath               # actually tell tex to use it!
    #    \usepackage{siunitx}    # micro symbols
    #    \sisetup{detect-all}    # force siunitx to use the fonts


#
#
# %% FORWARD OR BACKWARD
FORWARD_BACKWARD = 1  # 1=FORWARD, -1=BACKWARD


# %% 定数
RJ = float(7E+7)        # Jupiter半径   単位: m
MJ = float(1.90E+27)    # Jupiter質量   単位: kg
REU = float(1.56E+6)    # Europa半径    単位: m
RE = REU
MEU = float(4.8E+22)    # Europa質量    単位: kg

c = float(3E+8)         # 真空中光速    単位: m/s
me = float(9.1E-31)     # 電子質量      単位: kg
e = float(-1.6E-19)     # 電子電荷      単位: C

Gc = float(6.67E-11)    # 万有引力定数  単位: m^3 kg^-1 s^-2

mu = float(1.26E-6)     # 真空中透磁率  単位: N A^-2 = kg m s^-2 A^-2
Mdip = float(1.6E+27)   # Jupiterのダイポールモーメント 単位: A m^2
omgJ = FORWARD_BACKWARD*float(1.74E-4)    # 木星の自転角速度 単位: rad/s
omgE = FORWARD_BACKWARD*float(2.05E-5)  # Europaの公転角速度 単位: rad/s
omgR = omgJ-omgE             # 木星のEuropaに対する相対的な自転角速度 単位: rad/s


# %% 途中計算でよく出てくる定数の比など
# A1 = float(e/me)             # 運動方程式内の定数
# A2 = float(mu*Mdip/4/3.14)  # ダイポール磁場表式内の定数
A1 = float(-1.7582E+11)    # 運動方程式内の定数
A2 = FORWARD_BACKWARD*1.60432E+20            # ダイポール磁場表式内の定数
A3 = 4*3.1415*me/(mu*Mdip*e)    # ドリフト速度の係数


# %% Europa Position
lam = 10.0
REr = 9.6*RJ  # 木星からEuropa公転軌道までの距離

# 木星とtrace座標系原点の距離(x軸の定義)
R0 = REr*(np.cos(np.radians(lam)))**(-2)
R0x = R0
R0y = 0
R0z = 0

# Europaのtrace座標系における位置
eurx = REr*math.cos(math.radians(lam)) - R0
eury = 0
eurz = REr*math.sin(math.radians(lam))


# %% 初期座標ビンの設定(グローバル)
Nx, Ny, Nz = 20, 100, 1    # x: antijovian, y: corotation, z: north
Naeq = 140

# 初期条件座標エリアの範囲(最大と最小)
x_ip = (REr+1.25*RE)*(math.cos(math.radians(lam)))**(-2) - R0
x_im = (REr-1.25*RE)*(math.cos(math.radians(lam)))**(-2) - R0
y_ip = eury - 2*RE
y_im = 0

init_area = (x_ip-x_im)*(y_ip-y_im)*1*1E+6
N0 = (Nx*Ny*Nz*Naeq/init_area)  # 考えている粒子の磁気赤道面密度[cm^-3]


#
#
# %% 磁場
@jit('Tuple((f8,f8,f8))(f8,f8,f8)', nopython=True, fastmath=True)
def Bfield(x, y, z):
    # x, y, zは木星からの距離
    R2 = x**2 + y**2
    r_5 = math.sqrt(R2 + z**2)**(-5)

    Bx = A2*(3*z*x*r_5)
    By = A2*(3*z*y*r_5)
    Bz = A2*(2*z**2 - R2)*r_5

    return Bx, By, Bz


#
#
# %%
@jit(nopython=True, fastmath=True)
def maxwell(en):
    # en: 電子のエネルギー [eV]
    v = math.sqrt((en/me)*2*float(1.602E-19))

    # 中心 20eV
    kBT = 20  # eV
    kBT = kBT*(-e)
    fv20 = 4*np.pi*(v**2) * (me/(2*np.pi*kBT))**(1.5) * \
        np.exp(-(me*v**2)/(2*kBT))

    # 中心 300eV
    kBT = 300  # eV
    kBT = kBT*(-e)
    fv300 = 4*np.pi*(v**2) * (me/(2*np.pi*kBT))**(1.5) * \
        np.exp(-(me*v**2)/(2*kBT))

    fv = 0.95*fv20 + 0.05*fv300

    return fv


#
#
# %% 着地点における速度ベクトル(x, y, z成分)
def V(xyza, veq):
    # 木星原点に変換
    x = xyza[:, 0] + R0x
    y = xyza[:, 1] + R0y
    z = xyza[:, 2] + R0z
    aeq = xyza[:, 3]

    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)

    # Parallel
    nakami = 1 - r**5 * np.sin(aeq)**2 * \
        np.sqrt(r**2 + 3 * z**2) * (x**2 + y**2)**(-3)

    coef = veq * np.sqrt(nakami) * (r**(-1)) * (r**2 + 3 * z**2)**(-0.5)

    v_p_x = 3 * z * x
    v_p_y = 3 * z * y
    v_p_z = (2 * z**2) - (x**2 + y**2)

    v_par = np.stack([coef * v_p_x, coef * v_p_y, coef * v_p_z], 1)
    vpa2 = (coef**2) * (v_p_x**2 + v_p_y**2 + v_p_z**2)

    # Drift
    B = A2 * (np.sqrt(1+3*(z/r))**2) * r**(-3)
    Beq = A2 * np.sqrt(R0x**2 + R0y**2 + R0z**2)**(-3)
    vpe2 = (veq**2) * (B/Beq) * np.sin(aeq)**2
    theta = np.arccos(z/r)
    nakami = (vpa2) * r * np.sqrt(x**2 + y**2)
    nakami2 = 0.5 * vpe2 * (r**2) * np.sin(theta) * \
        (1 + (z/r)**2) * (1 + 3*(z/r)**2)**(-1)

    vb = A3 * (1+3*(z/r)**2)**(-1) * (nakami+nakami2)
    vb_x = np.zeros(x.shape)
    vb_y = omgR*x + vb
    vb_z = np.zeros(vb_x.shape)

    v_drift = np.stack((vb_x, vb_y, vb_z), 1)

    return v_par + v_drift


#
#
# %% 2つの3次元ベクトルの内積から、なす角を計算
@jit(nopython=True, fastmath=True)
def angle(A, B):
    # A, B... 3次元ベクトル

    # AベクトルとBベクトルの内積
    Dot = (A[:, 0]*B[:, 0] + A[:, 1]*B[:, 1] + A[:, 2]*B[:, 2])/(np.sqrt(A[:, 0] **
                                                                         2 + A[:, 1]**2 + A[:, 2]**2) * np.sqrt(B[:, 0]**2 + B[:, 1]**2 + B[:, 2]**2))

    # なす角
    ang = np.arccos(Dot)    # 単位: RADIANS

    return ang


#
#
# %% z軸周りの回転
@jit(nopython=True, fastmath=True)
def rot_xy(x, y, z, dp):
    xrot = x*np.cos(dp) + y*np.sin(dp)
    yrot = -x*np.sin(dp) + y*np.cos(dp)
    zrot = z
    rot = np.stack((xrot, yrot, zrot), axis=1)

    return rot


#
#
# %% 座標変換(Europa軌道座標系から木星dipole座標系に)
@jit(nopython=True, fastmath=True)
def rot_dipole(xyzad, lam):
    xi = xyzad[:, 0]
    xj = xyzad[:, 1]
    xk = xyzad[:, 2]

    xrot = xi*np.cos(np.radians(lam))+xk*np.sin(np.radians(lam))
    yrot = xj
    zrot = -xi*np.sin(np.radians(lam))+xk*np.cos(np.radians(lam))

    rot = np.stack((xrot, yrot, zrot, xyzad[:, 3], xyzad[:, 4]), axis=1)
    return rot


#
#
# %% 座標変換(Europa軌道座標系から木星dipole座標系に)
@jit(nopython=True, fastmath=True)
def rot_dipole2(xyz, lam):
    xi = xyz[0]
    xj = xyz[1]
    xk = xyz[2]

    xrot = xi*np.cos(np.radians(lam))+xk*np.sin(np.radians(lam))
    yrot = xj
    zrot = -xi*np.sin(np.radians(lam))+xk*np.cos(np.radians(lam))

    rot = np.array([xrot, yrot, zrot])
    return rot


#
#
# %% Europa中心座標に変換
@jit(nopython=True, fastmath=True)
def Europacentric(xyz, aeq):
    xyz[:, 0] += R0   # x座標原点を木星に
    rot = rot_dipole(xyz, lam)
    eur_rot = rot_dipole2(np.array([eurx+R0, eury, eurz]), lam)
    rot += - eur_rot  # Europa中心に原点を置き直す
    rxyza = np.stack((rot[:, 0], rot[:, 1], rot[:, 2], aeq), 1)   # aeq含む
    return rxyza


#
#
# %% 着地点の余緯度と経度を調べる
@jit(nopython=True, parallel=True)
def prefind(xyza):
    # 電子の座標(Europa centric)
    # x... anti jovian
    # y... corotation
    # z... northern

    x = xyza[:, 0]
    y = xyza[:, 1]
    z = xyza[:, 2]
    theta = np.arccos(z/np.sqrt(x**2+y**2+z**2))
    phi = np.arctan2(y, x)

    # phi... -pi to pi [RADIANS]
    # theta... 0 to pi [RADIANS]
    return np.stack((phi, theta), 1)


#
#
# %% 単位面積あたりのフラックスに直す
@jit(nopython=True, parallel=True)
def perarea(Xmesh, Ymesh, H):
    ntheta = int(300)
    dtheta = np.radians((Ymesh[1, 0]-Ymesh[0, 0])/ntheta)
    theta = np.radians(Ymesh[:-1, :-1])  # thetaのスタート

    nphi = int(300)
    dphi = np.radians((Xmesh[0, 1]-Xmesh[0, 0])/nphi)

    s = np.zeros(H.shape)  # ビンの面積
    for i in range(ntheta):
        s0 = np.sin(theta)
        theta += dtheta
        s1 = np.sin(theta)
        s += 0.5*(s0+s1)*dtheta

    for j in range(nphi):
        s += dphi

    s = s * REU**2

    # 単位面積あたりにする[m^-2]
    H = H/s

    return H


#
#
# %% マップ作成
def mapplot(maparray, x, vdotn, energy, denergy):
    # ヒストグラム作成時の重みづけ
    # nsin = 0
    # w = 1 / (omgR * (x + eurx + R0))
    # w = w * np.sin(aeq)**(nsin)
    w = vdotn

    # ヒストグラムの作成
    # k = int(1 + np.log2(216000))
    maparray = np.degrees(maparray)
    # xedges = list(np.linspace(-180, 180, 180))
    # yedges = list(np.linspace(0, 180, int(len(xedges)/2)))
    xedges = list(np.linspace(-180, 180, 80))
    yedges = list(np.linspace(0, 180, 40))

    H, xedges, yedges = np.histogram2d(
        maparray[:, 0], maparray[:, 1], bins=(xedges, yedges),
        # weights=w
    )
    H = H.T

    # メッシュの作成
    X, Y = np.meshgrid(xedges, yedges)

    # マクスウェル分布の計算
    fv = maxwell(energy)
    v1 = np.sqrt((energy-denergy)*2*(-e)/me)
    v2 = np.sqrt((energy)*2*(-e)/me)
    dv = v2 - v1
    f_dalpha = 1/60

    # dE dalpha を乗じた量に
    H = H*160*fv*dv*f_dalpha

    # Europaの位置における電子密度(Maxwell分布)
    zc = eurz + R0z
    n0 = 160*np.exp(-(zc/(1.7*RJ))**2)

    # 最大値と最小値
    print('======')
    print('H Max: ', np.max(H))
    print('H min: ', np.min(H))
    print('H ave: ', np.average(H))
    print('Maxwell [cc-1]: ', n0*fv*dv)
    print('======')

    # 単位面積あたりに変換[m^-3]
    # H = perarea(X, Y, H)

    # 単位[m^-2]を[cm^-2]に
    # H = H*1E-6

    # 物理量に[cm^-3 eV^-1]
    # H = H*(maxwell(energy)*160)/N0

    # x軸ラベル(標準的なwest longitude)
    xticklabels = ['360$^\\circ$W', '270$^\\circ$W',
                   '180$^\\circ$W', '90$^\\circ$W', '0$^\\circ$']

    # 図のタイトル
    title = 'colatitude-west longitude, $\\lambda=$' + \
        str(lam) + '$^\\circ$, $T=$ '+str(energy)+'eV'

    # 描画
    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.set_aspect(1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('West Longitude', fontsize=9)
    ax.set_ylabel('Colatitude', fontsize=9)
    ax.set_xticks(np.linspace(-180, 180, 5))
    ax.set_yticks(np.linspace(0, 180, 5))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(['0$^\\circ$', '45$^\\circ$', '90$^\\circ$',
                        '135$^\\circ$', '180$^\\circ$'])
    ax.invert_yaxis()
    mappable0 = ax.pcolormesh(X, Y, H, cmap='magma',
                              vmin=0)
    pp = fig.colorbar(mappable0, orientation='vertical')
    # pp.set_label('electrons s$^{-1}$ cm$^{-2}$', fontsize=10)   # フラックス
    pp.set_label('cm$^{-2}$ s$^{-1}$', fontsize=10)   # 密度
    fig.tight_layout()

    plt.show()

    """
    # 散布図でなんとかしたい
    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.set_aspect(1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('West Longitude', fontsize=9)
    ax.set_ylabel('Colatitude', fontsize=9)
    ax.set_xticks(np.linspace(-180, 180, 5))
    ax.set_yticks(np.linspace(0, 180, 5))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(['0$^\\circ$', '45$^\\circ$', '90$^\\circ$',
                        '135$^\\circ$', '180$^\\circ$'])
    ax.invert_yaxis()
    mappable0 = ax.scatter(
        maparray[:, 0], maparray[:, 1], s=0.2, c=w, cmap='magma')
    pp = fig.colorbar(mappable0, orientation='vertical')
    # pp.set_label('electrons s$^{-1}$ cm$^{-2}$', fontsize=10)   # フラックス
    pp.set_label('cm$^{-2}$ s$^{-1}$', fontsize=10)   # 密度
    fig.tight_layout()

    plt.show()
    """

    return 0


#
#
# %% HST比較用
def diskmap(rxyza, phi, obs):
    # rxyza... Europa centric の座標
    # x... anti jovian
    # y... orbital
    # z... northern
    # 0 x座標, 1 y座標, 2 z座標, 3 aeq
    # obs... sub-observer west longitude [DEGREES]

    # sub-obserber w long. をラジアンに変換
    obs = math.radians(obs)

    # phiを w long. に変換
    phi = np.abs(phi-np.pi)

    # sub-obs w long. が区間(270, 360)のとき
    if obs+0.5*np.pi > 2*np.pi:
        a = np.where(((phi >= 0) & (phi <= obs-1.5*np.pi)) |
                     ((phi >= obs-0.5*np.pi) & (phi < 2*np.pi)))
        print('270 to 360')

    # sub-obs w long. が区間[0, 90)のとき
    elif obs-0.5*np.pi < 0:
        a = np.where(((phi >= 0) & (phi <= obs+0.5*np.pi)) |
                     ((phi >= 1.5*np.pi+obs) & (phi < 2*np.pi)))
        print('0 to 90')

    # sub-obs w long. が区間[90, 270]のとき
    else:
        a = np.where((phi >= obs-0.5*np.pi) & (phi <= obs+0.5*np.pi))
        print('90 to 270')

    # 観測者から見える部分だけスライス
    rxyza = rxyza[a]

    # 座標の回転
    rxyzarot = rot_xy(rxyza[:, 0], rxyza[:, 1],
                      rxyza[:, 2], 1.5*np.pi-obs)  # xyzのみ
    rxyzarot = np.stack((rxyzarot[:, 0],    # 0 x座標
                         rxyzarot[:, 1],    # 1 y座標
                         rxyzarot[:, 2],    # 2 z座標
                         rxyza[:, 3]        # 3 磁気赤道面ピッチ角
                         ), axis=1)

    # 先行後行中心(meridian)
    colat = np.radians(np.linspace(1, 179, 50))  # 余緯度
    wlong = np.radians(np.array([90, 270]))      # 西経
    meshp, mesht = np.meshgrid(wlong, colat)  # meshp: 西経, mesht: 余緯度
    mesht = mesht.reshape(mesht.size)  # 1次元化
    meshp = meshp.reshape(meshp.size)  # 1次元化
    mrx = np.sin(mesht)*np.cos(meshp)
    mry = np.sin(mesht)*np.sin(meshp)
    mrz = np.cos(mesht)

    # 先行後行中心(meridian) 回転
    mrrot = rot_xy(mrx, mry, mrz, 1.5*np.pi-obs)
    mrrot = mrrot[np.where(mrrot[:, 1] <= 0)]  # 観測者から見える部分だけを抽出

    return rxyzarot, mrrot


#
#
# %%
def diskmapplot(rxyza, mrrot, obs, energy):
    # rxyza... 降り込み点 xyz座標 と aeq: shape = (*****, 4)
    # x-z平面に描画

    # ヒストグラム作成時の重みづけ
    aeq = rxyza[:, 3]
    w = 1 / (omgR * (rxyza[:, 0] + eurx + R0))
    rxyzaw = np.stack((
        rxyza[:, 0]/RE,    # 0: X'座標(Europa中心)(REで規格化)
        rxyza[:, 1]/RE,    # 1: Y'座標
        rxyza[:, 2]/RE,    # 2: Z'座標
        rxyza[:, 3],       # 3: aeq
        w,                    # 4: 重みづけ w
    ), axis=1)
    # w = w * 0.0031*np.exp(-((90-np.degrees(aeq))/22.5)**2)

    # ヒストグラム作成時の重みづけ
    aeq = rxyza[:, 3]

    # ヒストグラムの作成
    resolution = 36
    xedges = np.linspace(-2, 2, resolution)
    yedges = np.linspace(-2, 2, resolution)
    zedges = np.linspace(-2, 2, resolution)

    # 単位面積あたりに変換[m^-2]
    dY = yedges[1] - yedges[0]
    dZ = zedges[1] - zedges[0]

    H_disk = np.zeros((len(zedges)-1, len(xedges)-1))

    # 等経度面(X'一定のY'Z'面)を取り出す
    for i in range(len(xedges)-1):
        YZ_plane = rxyzaw[np.where(
            (rxyzaw[:, 0] >= xedges[i]) & (rxyzaw[:, 0] < xedges[i+1]))]
        print('YZ_plane shape: ', YZ_plane.shape)

        # 等経度面(y一定のxz面)でヒストグラム作成
        H, yedges, zedges = np.histogram2d(
            YZ_plane[:, 1], YZ_plane[:, 2], bins=(yedges, zedges),
            weights=YZ_plane[:, 4]
        )
        H = H.T
        print('H shape: ', H.shape)

        # メッシュの作成
        Y, Z = np.meshgrid(yedges, zedges)

        H = H / (dY * dZ * RE**2)

        # 単位[m^-3]を[cm^-3]に
        H = H*1E-6
        # print(H)

        # 密度分布をY方向に積分
        H_sight = np.sum(H, axis=1) * (RE*dY)
        print('H_sight shape: ', H_sight.shape)

        H_disk[:, i] = H_sight

    # print('H_disk: ', H_disk)

    """
    # 図のタイトル
    title = 'Y-Z plane | $\\varphi_{obs}=$' + \
        str(obs)+'$^\\circ$ | $\\lambda=$'+str(lam) + \
        '$^\\circ$ | $T=$'+str(energy)+'eV'

    # 描画
    fig, ax = plt.subplots(figsize=(6.25, 5))
    ax.set_aspect(1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('$Y\\ \\left( {\\rm R}_{\\rm E} \\right)$', fontsize=9)
    ax.set_ylabel('$Z\\ \\left( {\\rm R}_{\\rm E} \\right)$', fontsize=9)
    ax.invert_xaxis()
    # ax.set_xlim([-1.5, 1.5])
    # ax.set_ylim([-1.5, 1.5])
    # ax.plot(mrrot[:, 0], mrrot[:, 2], color='#FFFFFF',
    #         linestyle='dashed')   # 先行後行 meridian
    # ax.plot(np.array([-1, 1]), np.array([0, 0]),
    #         color='#FFFFFF')  # Europa赤道
    # ax.add_patch(plt.Circle((0, 0), 1, color='#FFFFFF',
    #                       fill=False, alpha=1.0))  # Europaディスク(中心座標と半径を指定)

    # カラーマップの描画
    mappable0 = ax.pcolormesh(Y, Z, H, cmap='magma',
                              vmin=0)
    pp = fig.colorbar(mappable0, orientation='vertical', ax=ax)
    pp.set_label('electrons s$^{-1}$ cm$^{-2}$', fontsize=10)

    fig.tight_layout()

    plt.show()
    """

    # ディスク描画
    # メッシュの作成
    X, Z = np.meshgrid(xedges, zedges)
    X_lim = np.where((X[:-1, :-1] < -0.3) | (X[:-1, :-1] > 0.3))
    Z_lim = np.where((Z[:-1, :-1] < -0.3) | (Z[:-1, :-1] > 0.3))
    # H_disk[X_lim] = 0
    # H_disk[Z_lim] = 0

    # 図のタイトル
    title = 'X-Z plane | $\\varphi_{obs}=$' + \
        str(obs)+'$^\\circ$ | $\\lambda=$'+str(lam) + \
        '$^\\circ$ | $T=$'+str(energy)+'eV'

    # 描画
    fig, ax = plt.subplots(figsize=(6.25, 5))
    ax.set_aspect(1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('$X\\ \\left( {\\rm R}_{\\rm E} \\right)$', fontsize=9)
    ax.set_ylabel('$Z\\ \\left( {\\rm R}_{\\rm E} \\right)$', fontsize=9)
    ax.plot(mrrot[:, 0], mrrot[:, 2], color='#FFFFFF',
            linestyle='dashed')   # 先行後行 meridian
    ax.plot(np.array([-1, 1]), np.array([0, 0]),
            color='#FFFFFF')  # Europa赤道
    ax.add_patch(plt.Circle((0, 0), 1, color='#FFFFFF',
                            fill=False, alpha=1.0))  # Europaディスク(中心座標と半径を指定)

    # カラーマップの描画
    mappable0 = ax.pcolormesh(X, Z, H_disk, cmap='magma',
                              vmin=0)
    pp = fig.colorbar(mappable0, orientation='vertical', ax=ax)
    pp.set_label('electrons cm$^{-3}$', fontsize=10)

    fig.tight_layout()

    plt.show()

    return 0


#
#
# %% main関数
def main():
    start = time.time()  # 時間計測開始

    # エネルギー一覧
    enlist = list([10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                  200, 300, 400, 500, 700, 1000,
                  2000, 3000, 4000, 5000, 7000, 10000,
                  20000])
    devlist = list([10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                   100, 100, 100, 100, 200, 300,
                   1000, 1000, 1000, 1000, 2000, 3000,
                   10000])

    # エネルギー
    energy = 10
    denergy = 10
    # veq = math.sqrt((energy/me)*2*float(1.602E-19))

    # ファイル読み込み
    filepath0 = '/Users/satoshin/Library/Mobile Documents/com~apple~CloudDocs/PPARC/gc203g_' + \
        str(energy)+'ev_20220109_1.txt'

    # 座標&ピッチ角ファイル
    a0 = np.loadtxt(filepath0)
    # a0[:, 0] ... 出発点 x座標
    # a0[:, 1] ... 出発点 y座標
    # a0[:, 2] ... 出発点 z座標
    # a0[:, 3] ... 終点 x座標
    # a0[:, 4] ... 終点 y座標
    # a0[:, 5] ... 終点 z座標
    # a0[:, 6] ... yn (=1)
    # a0[:, 7] ... 終点 energy [eV]
    # a0[:, 8] ... 終点 alpha_eq [RADIANS]
    # a0[:, 9] ... 出発点 v_dot_n

    print(np.where(np.isnan(a0).any(axis=1)))
    print(a0[np.where(np.isnan(a0).any(axis=1))])
    # a0 = a0[np.where(np.isnan(a0).any(axis=1))]

    a0 = a0[np.where(a0[:, 9] < 0)]  # vdotn < 0 のみが適切
    # print('vdotn: ', a0[:, 9])

    # 表面着地点座標 & ピッチ角 (検索は表面y座標=1列目)
    xyz_surf = a0[:, 0:3]
    aeq_surf = a0[:, 8]
    vdotn = a0[:, 9]

    print('execution time: %.3f sec' % (time.time() - start))  # 計算時間表示

    # 描画
    # 表面密度分布 緯度経度
    rxyza = Europacentric(xyz_surf, aeq_surf)
    surf_map = prefind(rxyza)   # aeq含まず
    print('data shape: ', a0.shape)
    print('surface count: {:>7d}'.format(xyz_surf[:, 0].size))
    mapplot(surf_map, xyz_surf[:, 0], -vdotn, energy, denergy)

    return 0


#
#
# %%
if __name__ == '__main__':
    a = main()
