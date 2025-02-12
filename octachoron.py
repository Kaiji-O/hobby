import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation

print('方向ベクトル(x, y, z, u)を入力してください。(u>0。他の要素は非負。半角スペース区切り。)')
e = np.array(list(map(float, input().split())))
ex = np.array([1, 0, 0, 0])
ey = np.array([0, 1, 0, 0])
ez = np.array([0, 0, 1, 0])
eu = np.array([0, 0, 0, 1])
basis = [ex, ey, ez, eu]

# 頂点
class Node:
    def __init__(self, x, y, z, u):
        self.r = np.array([x, y, z, u])

# 辺
# node1→node2のような方向を持っていると考える。
class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.dir = node2.r - node1.r

# 点p1とp2が同一面上にあるかチェックする関数
def is_on_same_face(p1, p2):
    c = 0
    for b in basis:
        if np.dot(b, p1-p2) == 0:
            c += 1
    if c >= 2:
        return True
    else:
        return False

def draw_line(ax, p1, p2, im):
    line = art3d.Line3D([p1[0], p2[0]],[p1[1], p2[1]],[p1[2], p2[2]], color='c')
    im.append(ax.add_line(line))

# 正規化
def normalize(v):
    return v / np.linalg.norm(v, ord=2)

# Gram-Schmidtの直交化
def gram_schmidt(e):
    eX = ex - np.dot(e, ex) * e
    eX = normalize(eX)
    eY = ey - np.dot(e, ey) * e - np.dot(eX, ey) * eX
    eY = normalize(eY)
    eZ = ez - np.dot(e, ez) * e - np.dot(eX, ez) * eX - np.dot(eY, ez) * eY
    eZ = normalize(eZ)
    return (eX, eY, eZ)

node_list = []
edge_list = []

for i in range(16):
    # 頂点を登録
    x = i//8
    y = (i - 8*x) // 4
    z = (i - 8*x - 4*y) // 2
    u = i - 8*x - 4*y - 2*z
    node = Node(x, y, z, u)
    node_list.append(node)
    # 辺を登録
    # node1が(*, *, *, 0)、node2が(*, *, *, 1)のようになるようにする。
    if x == 1:
        edge = Edge(node_list[i-8], node)
        edge_list.append(edge)
    if y == 1:
        edge = Edge(node_list[i-4], node)
        edge_list.append(edge)
    if z == 1:
        edge = Edge(node_list[i-2], node)
        edge_list.append(edge)
    if u == 1:
        edge = Edge(node_list[i-1], node)
        edge_list.append(edge)

e = normalize(e)
eX, eY, eZ = gram_schmidt(e)

fig = plt.figure(figsize=(10, 7),dpi=120)
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1,1,1))

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ims = []

for t in np.linspace(0, 2, 200):
    intersection_list = []
    projected_intersection_list = []

    im = []

    for edge in edge_list:
        # 交点を計算し、登録
        if np.dot(e, edge.dir) == 0:
            continue
        a = (t - np.dot(e, edge.node1.r)) / np.dot(e, edge.dir)
        if 0 <= a <= 1:
            intersection_list.append(a * edge.dir + edge.node1.r)

    for p in intersection_list:
        # 交点を直交補空間上の座標に変換、登録
        x = np.dot(p-t*e, eX)
        y = np.dot(p-t*e, eY)
        z = np.dot(p-t*e, eZ)
        projected_intersection_list.append(np.array([x, y, z]))

    for i in range(len(intersection_list)):
        for j in range(i + 1, len(intersection_list)):
            # 2点が同一面上にある場合は線分で結ぶ
            if is_on_same_face(intersection_list[i], intersection_list[j]):
                draw_line(ax, projected_intersection_list[i], projected_intersection_list[j], im)
    ims.append(im)

ani = animation.ArtistAnimation(fig, ims, interval=10)
plt.show()
