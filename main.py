import random
import numpy as np
import cv2 as cv
import time

# 1. T = text, image, Laplace equation, fragment of the code (1 point)
# 2. Convert T in X: (2 points)
# 2.1 entries of X are integer numbers
# 2.2 entries of X are floating point numbers
# 3. Encript X and obtain Y : (2points)
# 3.1 Set key matrix K, explain why you you decided for this K
# 3.2 Consider digital image or part of it as a key matrix K.
# 3.3 Using Gram-Schmidt or modified Gram-Schmidt algorithm check if K is
# invertible.
# 4. Set digital image Z and hide Y in Z using LSB approach. Z˜ be image
# containing Y (1 point)
# 5. Recover Y from Z˜ (1 point)
# 6. Decript Y and obtain X using: (3 points)
# 6.1 classic iterative methods
# 6.2 Richardson’s method with preconditioner
# 6.3 Non-stationary interative method
# 6.4 Justify termination criteria used in computations
# 7. Recover T from X with integer and digital entries (2points)

# Global Variables
VEC1 = []
VEC2 = []

DEC1 = []
DEC2 = []

# RICHARDSON
# P (x^k+1 - x^k) / Tk + Ax = b

def Richardson(A, b, P, Tk = 1, iterations=2, TOL=0.0001):
    print(A)
    B = np.eye(A.shape[0]) - np.matmul(np.linalg.inv(P), A)
    print(b,"b")
    f = np.matmul(np.linalg.inv(P),b)
    x0 = np.zeros_like(b)
    x = x0.copy()
    for iter in range(iterations):
        xN = np.matmul(B,x) + f
        if np.linalg.norm((x - xN),ord=2) < TOL:
            x = xN.copy()
            break
        x = xN.copy()
        print(x)
    return x


# GRAM SCHMIDT FOR DETECTING IF THE MATRIX IS DEGENERATE OR NOT

def projection(u, v):
    res = u * np.dot(u, v) / np.dot(u, u)
    return res


def project(V, k, uVecs, n):
    ki = k-1

    projSum = np.array([0] * n, dtype=np.float64)
    while ki >= 0:
        # print(ki, uVecs[ki],V)
        projSum += projection(uVecs[ki], V)
        ki -= 1
    return projSum


def checkDegeneracy(vecs):
    degenerate = False
    siz = len(vecs)
    for i in range(siz):
        for j in range(i+1, siz):
            if checkConvergence(vecs[i], vecs[j]) < 0.0001:
                degenerate = True

    return degenerate


def gramSchmidt(vecs, vecAmount, n):
    uVecs = []
    eVecs = []
    projVec = np.array([0] * n)
    uVecs.append(vecs[0])
    matrixDumb = False
    for i in range(0, vecAmount):
        Vi = np.array(vecs[i])
        u = Vi - projVec
        uVecs.append(u)

        uv = np.array(uVecs)
        nrm = np.linalg.norm(u, ord=2)
        if (nrm <= 0):
            matrixDumb = True
            break
        else:
            ev = u / nrm
            eVecs.append(ev)
        projVec = project(Vi, i, uv, n)

    matrixDumb = checkDegeneracy(eVecs)
    return matrixDumb


# --------------------------------------

# --------------------------------------

# ITERATIVE METHOD I CHOSE - SOR
def infNorm(vec):
    res = 0
    for el in vec:
        res = max(abs(el), res)
    return res


def checkConvergence(a, b):
    size = len(a)
    c = []
    for i in range(size):
        c.append(a[i] - b[i])
    return infNorm(c)


def SOR(A, B, x0, w, tolerance, n, N):
    debugger = True  # Toggle for iteration info
    Xvec = x0.copy()
    iteration = 1
    while (iteration <= N):
        for i in range(n):
            sm = 0
            for j in range(n):
                if j == i:
                    continue
                sm -= A[i][j] * Xvec[j]
            sm += B[i]
            Xvec[i] = round((1-w) * Xvec[i] + w * sm / A[i][i])
        if (debugger):
            if (checkConvergence(Xvec, x0) < tolerance):
                # print("------------------")
                # print("Tolerance achieved")
                break

        x0 = Xvec.copy()
        iteration += 1
    return Xvec

# --------------------

# Function has a mode parameter


def genKeyText(chunkSize, rowCount, originalSize):
    mat = []
    for row in range(chunkSize):
        rowList = []
        for col in range(chunkSize):
            if row == 0 and col == 0:
                rowList.append(chunkSize)
            elif row == 1 and col == 1:
                rowList.append(rowCount)
            elif row == 2 and col == 2:
                rowList.append(originalSize)
            elif col == 0:
                rowList.append(row - 1)
            elif row == col:
                rowList.append((2 * chunkSize - row) % 500 + 1)
            else:
                rowList.append(random.randint(0, 2))
        mat.append(rowList)

    res = gramSchmidt(mat, chunkSize, chunkSize)
    if res:
        print("YOU HAVE CHOSEN A BAD KEY MATRIX TRY AGAIN")
        mat = genKeyText(chunkSize, rowCount, originalSize)
    else:
        print("You have chosen a very good Key matrix")
        matrix = np.array(mat)
        return matrix


def genKeyForImage(chunkSize, rowCount, dimensions):
    mat = []
    rowTh = int(rowCount / 1000)
    rowNoTh = rowCount % 1000
    thousands = int(dimensions[2] / 1000)
    noThousand = int(dimensions[2] % 1000)
    for row in range(chunkSize):
        rowList = []
        for col in range(chunkSize):
            if row == 0 and col == 0:
                rowList.append(chunkSize)
            elif row == 2 and col == 2:
                # Width
                rowList.append(dimensions[0])
            elif row == 3 and col == 3:
                # Height
                rowList.append(dimensions[1])
            elif row == 4 and col == 4:
                # originalFlatSize
                rowList.append(thousands)
            elif row == 5 and col == 5:
                rowList.append(noThousand + 1)
            elif row == 6 and col == 6:
                rowList.append(rowTh + 100)
            elif row == 7 and col == 7:
                rowList.append(rowNoTh + 1)
            elif row == col:
                rowList.append(20)
            elif row < col:
                rowList.append(random.randint(0, 2))
            else:
                rowList.append(0)
        # print(rowList)
        mat.append(rowList)

    res = gramSchmidt(mat, chunkSize, chunkSize)
    if res:
        print("YOU HAVE CHOSEN A BAD KEY MATRIX TRY AGAIN")
        mat = genKeyForImage(chunkSize, rowCount, dimensions)
    else:
        print("You have chosen a very good Key matrix")
        matrix = np.array(mat)
        return matrix


def encrypt(mode, input, secImage):
    if mode == 'text':

        # Do text stuff
        t = input
        chunkSize = 15

        # Turn chars into ascii numbers and divide in chunks
        tVec = list(map(lambda el: ord(el), t))
        remainder = len(t) % chunkSize

        # Add elements if remainder is not zero (Here are 3 cases that may happen):
        # a a a
        # b b b
        # c c   -> rem = 8 % 3 = 2 -> add 3 - 2 = 1
        # -----------------------------------------
        # a a -> rem = 2 % 3 = 2 -> add 3 - 2 = 1
        # -----------------------------------------
        # a a a -> rem = 3 % 3 = 0 -> add 3 - 0 = 3 ? No, don't add

        originalSize = len(tVec)
        if remainder != 0:
            for i in range(chunkSize - remainder):
                tVec.append(32)
            inpVec = np.array(tVec)
            rowCount = int(len(inpVec) / chunkSize)
            inpVec = inpVec.reshape(rowCount, chunkSize)

        # Generate key matrix
        key = genKeyText(chunkSize, rowCount, originalSize)

        yMat = []
        for r in range(rowCount):
            yVec = np.matmul(key, inpVec[r])
            yMat.append(yVec)
        yMat = np.array(yMat)

        ybin = ''
        for r in range(yMat.shape[0]):
            ybin += ''.join(list(map(lambda el: format(el, '032b'), yMat[r])))

        # With LSB approach hide bits of ybin into the secImage
        width = secImage.shape[0]
        height = secImage.shape[1]
        secImage = secImage.flatten()
        for i, bit in enumerate(ybin):
            v = secImage[i]
            v = bin(v)
            v = v[:-1] + bit
            secImage[i] = int(v, 2)
        secImage = secImage.reshape((width, height, 3))
        return (secImage, key)

    elif mode == "image":

        # Do image stuff
        t = input
        chunkSize = 15

        originalWidth = t.shape[0]
        originalHeight = t.shape[1]

        t = t.flatten()
        flatSize = len(t)
        tList = list(t)
        remainder = flatSize % chunkSize
        originalFlatSize = flatSize

        # Add elements if remainder is not zero (Here are 3 cases that may happen):
        # a a a
        # b b b
        # c c   -> rem = 8 % 3 = 2 -> add 3 - 2 = 1
        # -----------------------------------------
        # a a -> rem = 2 % 3 = 2 -> add 3 - 2 = 1
        # -----------------------------------------
        # a a a -> rem = 3 % 3 = 0 -> add 3 - 0 = 3 ? No, don't add

        if remainder != 0:
            if flatSize < chunkSize:
                for i in range(chunkSize - remainder):
                    tList.append(0)
            elif flatSize > chunkSize:
                for i in range(chunkSize - remainder):
                    tList.append(0)

        inpVec = np.array(tList)
        rowCount = int(flatSize // chunkSize)

        inpVec = inpVec.reshape(rowCount, chunkSize)

        # Generate Key Matrix
        key = genKeyForImage(chunkSize, rowCount,
                             (originalWidth, originalHeight, originalFlatSize))

        yMat = []
        for r in range(rowCount):
            yVec = np.matmul(key, inpVec[r])
            yMat.append(yVec)

        yMat = np.array(yMat)

        global DEC1
        DEC1 = yMat.flatten()

        ybin = ''
        yInts = []
        for r in range(yMat.shape[0]):
            b = ''.join(list(map(lambda el: format(el, '032b'), yMat[r])))
            ybin += b

        global VEC1
        VEC1 = ybin[:originalFlatSize * 16]

        aa = yMat[0][0]
        # print(aa)

        a = ybin[0:32]

        # With LSB approach hide bits of ybin into the secImage
        width = secImage.shape[0]
        height = secImage.shape[1]

        secImage = secImage.flatten()
        yindex = 0
        index = 0
        while yindex < len(ybin):
            v = secImage[index]
            v = bin(v)
            vSize = len(v)
            v = v[0:vSize - 2] + ybin[yindex:yindex + 2]
            secImage[index] = int(v, 2)
            yindex += 2
            index += 1
        secImage = secImage.reshape((width, height, 3))
        return (secImage, key)


def Diagonal(A):
    D = []
    rowSize = A.shape[0]
    for i in range(rowSize):
        rowList = []
        for c in range(rowSize):
            if i == c:
                rowList.append(2)
            else:
                rowList.append(0)
        D.append(rowList)
    D = np.array(D)
    return D


def decrypt(mode, key, secImage):
    if mode == "text":
        text = []
        index = 0
        secImage = secImage.flatten()
        chunkSize = key[0][0]
        rowCount = key[1][1]
        originalSize = key[2][2]
        goalSize = rowCount * chunkSize
        while index < secImage.shape[0] and len(text) < goalSize:
            bits = ''.join([bin(i)[-1] for i in secImage[index:index + 32]])
            text.append(int(bits, 2))
            index += 32

        result = ''
        for r in range(rowCount):
            startIndex = r * chunkSize
            endIndex = startIndex + chunkSize
            t = np.array(text[startIndex:endIndex])
            X = SOR(key, t, np.array([0] * t.shape[0]),
                    1., 0.0001, t.shape[0], 100)
            result += ''.join([chr(abs(i)) for i in X])

        result = result[:originalSize]
        return result

    if mode == "image":
        allVals = []
        index = 0
        secImage = secImage.flatten()
        chunkSize = key[0][0]
        rowCount = (key[6][6]-100)*1000 + (key[7][7] - 1)
        originalWidth = key[2][2]
        originalHeight = key[3][3]
        originalFlatSize = key[4][4] * 1000 + (key[5][5] - 1)
        print(originalFlatSize)
        goalSize = rowCount * chunkSize

        elSize = len(bin(secImage[0]))
        ybin = ''
        while index < secImage.shape[0] and len(allVals) < goalSize:
            bits = ''.join([bin(i)[elSize-2:elSize]
                           for i in secImage[index:index+16]])
            ybin += bits
            allVals.append(int(bits, 2))
            index += 16

        global DEC2
        DEC2 = allVals

        global VEC2
        VEC2 = ybin

        result = []
        for r in range(rowCount):
            if len(result) < originalFlatSize:
                startIndex = r * chunkSize
                endIndex = startIndex + chunkSize
                t = np.array(allVals[startIndex: endIndex])
                X = SOR(key, t, np.array([0]*t.shape[0]),
                        1., 0.0001, t.shape[0], 30)
                X = list(X)
                result.append(X)
            else:
                break

        result = np.array(result)
        result = result.reshape((originalWidth, originalHeight, 3))
        return np.uint8(result)


def checker(mode, input, secImage):
    if mode == 'text':
        inpSize = len(input)
        bitUseCount = 1
        vals4one = 32 / bitUseCount
        canStoreAmount = int(
            secImage.shape[0] * secImage.shape[1] * 3 / vals4one)
        if inpSize > canStoreAmount:
            raise ValueError("Can't store more than",
                             canStoreAmount, "input size")
        else:
            print("Your input size:", len(input),
                  "| Max Input size:", str(canStoreAmount))
    elif mode == 'image':
        bitUseCount = 2
        vals4one = 32 / bitUseCount
        hideAmount = int(img.shape[0] * img.shape[1] * 3)
        canStoreAmount = int(
            secImage.shape[0] * secImage.shape[1] * 3 / vals4one)
        mxInp = max(input.shape[0], input.shape[1])
        mxCover = min(secImage.shape[0], secImage.shape[1])
        if (mxInp * 10 > mxCover):
            raise ValueError(
                "Max dimension of hidden image should be ten times smaller than min dimension of cover image, sincerely Luka xD")
        if (hideAmount > canStoreAmount):
            raise ValueError("Can't store more than "
                             + str(int(canStoreAmount / 10)) + " input size")
        else:
            print("Your input size:", hideAmount,
                  "| Max Input size:", str(canStoreAmount))


img = cv.imread('images/Blue200.png')
secImage = cv.imread('images/Red2000.png')
text = """Some random text 123 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa xDDD
            eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
            heheeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
            sheeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeesh"""

mode = str(input("Choose program mode --- text/image--- "))
inp = ''
if mode == "text":
    inp = text
elif mode == "image":
    inp = img

# Check if input can be hidden
checker(mode, inp, secImage)

start = time.time()
encImg, key = encrypt(mode, inp, secImage)
decripted = decrypt(mode, key, encImg)
end = time.time()
print("----------------------------")
print("Time needed to calculate:", end - start)
print("----------------------------")
cv.imshow("Encrypted Image", encImg)
if mode == "text":
    print("Decripted | ", "Length:", len(decripted))
    print(decripted)

cv.waitKey(0)
cv.destroyAllWindows()

if mode == "image":
    cv.imshow("Dncrypted Image", decripted)
    cv.waitKey(0)
    cv.destroyAllWindows()

