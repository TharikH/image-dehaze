def airlight(img,kernel_size):
    b,g,r = cv2.split(img)
    size=(kernel_size,kernel_size)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape,size)
    r_max = cv2.dilate(r,kernel)
    g_max = cv2.dilate(g,kernel)
    b_max = cv2.dilate(b,kernel)
    r_avg=np.sum(a1)/np.size(a1)
    g_avg=np.sum(a2)/np.size(a2)
    b_avg=np.sum(a3)/np.size(a3)
    A=[r_avg,g_avg,b_avg]
    return A