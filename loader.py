def load_images():
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    from skimage import color
    import numpy as np
    road1 = shuffle(plt.imread("data/road1.jpg").reshape(-1,3))[:800]
    road2 = shuffle(plt.imread("data/road2.jpg").reshape(-1, 3))[:800]
    road3 = shuffle(plt.imread("data/road3.jpg").reshape(-1, 3))[:150]
    unitedroad=shuffle(plt.imread("data/unitedroad.jpg").reshape(-1,3))[:4000]
    notroad1 = shuffle(plt.imread("data/notroad1.jpg").reshape(-1, 3))[:1000]
    notroad2 = shuffle(plt.imread("data/notroad2.jpg").reshape(-1, 3))[:700]
    tree1 = shuffle(plt.imread("data/tree1.jpg").reshape(-1,3))[:600]
    tree2 = shuffle(plt.imread("data/tree2.jpg").reshape(-1,3))[:500]
    tree3 = shuffle(plt.imread("data/tree3.jpg").reshape(-1, 3))[:640]
    # print tree.shape
    # print tree2.shape
    # tree = shuffle(plt.imread("data/tree.jpg")).reshape(-1,3)
    data_road = np.concatenate((road1,unitedroad, road2,road3,notroad1, notroad2,tree1,tree2,tree3))
    data_road=data_road.reshape(-1,1,3)
    labels_road = np.concatenate((np.array([1 for i in xrange(1750)] + [1 for i in xrange(4000)]), np.array([2 for j in xrange(1700)]),
                                  np.array([3 for k in xrange(1740)])))

    return data_road, labels_road

def load_imagesforcnn():
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    import numpy as np
    # road1 = shuffle(plt.imread("data/road1.jpg"))[:800]
    # road2 = shuffle(plt.imread("data/road2.jpg"))[:800]
    # road3 = shuffle(plt.imread("data/road3.jpg"))[:150]
    # notroad1 = shuffle(plt.imread("data/notroad1.jpg"))[:1000]
    # notroad2 = shuffle(plt.imread("data/notroad2.jpg"))[:700]
    # tree1 = shuffle(plt.imread("data/tree1.jpg"))[:600]
    # tree2 = shuffle(plt.imread("data/tree2.jpg"))[:500]
    tree3 = shuffle(plt.imread("data/tree3.jpg"))[:640]
    # print tree.shape
    # print tree2.shape
    # tree = shuffle(plt.imread("data/tree.jpg")).reshape(-1,3)
    print tree3.shape

    labels_road = np.concatenate((np.array([1 for i in xrange(1750)]), np.array([2 for j in xrange(1700)]),
                                  np.array([3 for k in xrange(1740)])))

    return labels_road

def image_data():
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    import numpy as np
    road=plt.imread("data/")

def hsv_images():
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    from skimage import color
    import numpy as np
    road1 = shuffle(color.rgb2hsv(plt.imread("data/road1.jpg")).reshape(-1,3))[:800]
    road2 = shuffle(color.rgb2hsv(plt.imread("data/road2.jpg")).reshape(-1, 3))[:800]
    road3 = shuffle(color.rgb2hsv(plt.imread("data/road3.jpg")).reshape(-1, 3))[:150]
    road4 = shuffle(color.rgb2hsv(plt.imread("data/notroad.jpg")).reshape(-1, 3))[:6000]
    unitedroad=shuffle(color.rgb2hsv(plt.imread("data/unitedroad.jpg")).reshape(-1,3))[:4000]

    notroad1 = shuffle(color.rgb2hsv(plt.imread("data/notroad1.jpg")).reshape(-1, 3))[:1000]
    notroad2 = shuffle(color.rgb2hsv(plt.imread("data/notroad2.jpg")).reshape(-1, 3))[:700]
    notroad3 = shuffle(color.rgb2hsv(plt.imread("data/Untitled90.jpg")).reshape(-1, 3))[:8000]

    tree1 = shuffle(color.rgb2hsv(plt.imread("data/tree1.jpg")).reshape(-1,3))[:600]
    tree2 = shuffle(color.rgb2hsv(plt.imread("data/tree2.jpg")).reshape(-1,3))[:500]
    tree3 = shuffle(color.rgb2hsv(plt.imread("data/tree3.jpg")).reshape(-1, 3))[:640]
    # print tree.shape
    # print tree2.shape
    # tree = shuffle(plt.imread("data/tree.jpg")).reshape(-1,3)
    data_road = np.concatenate((road1,unitedroad, road2,road3,road4,notroad1, notroad2,notroad3,tree1,tree2,tree3))

    labels_road = np.concatenate((np.array([1 for i in xrange(1750+6000)] + [1 for i in xrange(4000)]), np.array([2 for j in xrange(1700+8000)]),
                                  np.array([3 for k in xrange(1740)])))

    return data_road, labels_road

def hsvrgb_images():
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    from skimage import color
    import numpy as np
    a=plt.imread("data/road1.jpg").reshape(-1,3)
    b=color.rgb2hsv(plt.imread("data/road1.jpg")).reshape(-1, 3)
    road1=shuffle(np.concatenate((a,b),axis=1))[:800]
    c=color.rgb2hsv(plt.imread("data/road2.jpg")).reshape(-1,3)
    d=plt.imread("data/road2.jpg").reshape(-1,3)
    road2 = shuffle(np.concatenate((c, d), axis=1))[:800]
    e=plt.imread("data/road3.jpg").reshape(-1,3)
    f=color.rgb2hsv(plt.imread("data/road3.jpg")).reshape(-1,3)
    road3=shuffle(np.concatenate((e,f),axis=1))[:150]
    g=plt.imread("data/notroad.jpg").reshape(-1,3)
    h=color.rgb2hsv(plt.imread("data/notroad.jpg")).reshape(-1,3)
    road4=shuffle(np.concatenate((g,h),axis=1))[:6000]
    i=plt.imread("data/unitedroad.jpg").reshape(-1,3)
    j=color.rgb2hsv(plt.imread("data/unitedroad.jpg")).reshape(-1,3)
    unitedroad=shuffle(np.concatenate((i,j),axis=1))[:4000]
    # road2 = shuffle(plt.imread("data/road2.jpg").reshape(-1, 3))[:800]
    # road3 = shuffle(plt.imread("data/road3.jpg").reshape(-1, 3))[:150]
    # road4 = shuffle(plt.imread("data/notroad.jpg").reshape(-1, 3))[:6000]
    # unitedroad=shuffle(plt.imread("data/unitedroad.jpg").reshape(-1,3))[:4000]

    k=plt.imread("data/notroad1.jpg").reshape(-1,3)
    l=color.rgb2hsv(plt.imread("data/notroad1.jpg")).reshape(-1,3)
    notroad1 = shuffle(np.concatenate((k,l),axis=1))[:1000]
    m=plt.imread("data/notroad2.jpg").reshape(-1,3)
    n=color.rgb2hsv(plt.imread("data/notroad2.jpg")).reshape(-1,3)
    notroad2=shuffle(np.concatenate((m,n),axis=1))[:700]
    o=plt.imread("data/Untitled90.jpg").reshape(-1,3)
    p=color.rgb2hsv(plt.imread("data/Untitled90.jpg")).reshape(-1,3)
    notroad3=shuffle(np.concatenate((o,p),axis=1))[:8000]
    # notroad2 = shuffle(plt.imread("data/notroad2.jpg").reshape(-1, 3))[:700]
    # notroad3 = shuffle(plt.imread("data/Untitled90.jpg").reshape(-1, 3))[:8000]

    q=plt.imread("data/tree1.jpg").reshape(-1,3)
    r=color.rgb2hsv(plt.imread("data/tree1.jpg")).reshape(-1,3)
    tree1=shuffle(np.concatenate((q,r),axis=1))[:600]
    s=plt.imread("data/tree2.jpg").reshape(-1,3)
    t=color.rgb2hsv(plt.imread("data/tree2.jpg")).reshape(-1,3)
    tree2=shuffle(np.concatenate((s,t),axis=1))[:500]
    u=plt.imread("data/tree3.jpg").reshape(-1,3)
    v=color.rgb2hsv(plt.imread("data/tree3.jpg")).reshape(-1,3)
    tree3=shuffle(np.concatenate((u,v),axis=1))[:640]

    # tree1 = shuffle(plt.imread("data/tree1.jpg").reshape(-1,3))[:600]
    # tree2 = shuffle(plt.imread("data/tree2.jpg").reshape(-1,3))[:500]
    # tree3 = shuffle(plt.imread("data/tree3.jpg").reshape(-1, 3))[:640]
    # print tree.shape
    # print tree2.shape
    # tree = shuffle(plt.imread("data/tree.jpg")).reshape(-1,3)
    data_road = np.concatenate((road1,unitedroad, road2,road3,road4,notroad1, notroad2,notroad3,tree1,tree2,tree3))

    labels_road = np.concatenate((np.array([1 for i in xrange(1750+6000)] + [1 for i in xrange(4000)]), np.array([2 for j in xrange(1700+8000)]),
                                  np.array([3 for k in xrange(1740)])))

    return data_road, labels_road













