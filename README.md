## Documentation CP - 2

### Problem:
1. T = text, image, Laplace equation, fragment of the code (1 point)
2. Convert T in X: (2 points)
2.1 entries of X are integer numbers
2.2 entries of X are floating point numbers
3. Encript X and obtain Y : (2points)
3.1 Set key matrix K, explain why you you decided for this K
3.2 Consider digital image or part of it as a key matrix K.
3.3 Using Gram-Schmidt or modified Gram-Schmidt algorithm check if K is
 invertible.
4. Set digital image Z and hide Y in Z using LSB approach. Z˜ be image
containing Y (1 point)
5. Recover Y from Z˜ (1 point)
6. Decript Y and obtain X using: (3 points)
6.1 classic iterative methods
6.2 Richardson’s method with preconditioner
6.3 Non-stationary interative method
6.4 Justify termination criteria used in computations
7. Recover T from X with integer and digital entries (2points)

### My Solution:
1) Program works for all inputs (PS. Secret image must be 10 times smaller than cover image for best results)
2) I convert input to one dimensional int array
3) I choose random diagonally dominant matrix for key and store dimensions of images inside, i check if the matrix is invertible with the help of gram schmidt algorithm. My version of gram schmidt gives me basises and i check them if they're linearly dependent or not
4) I hide Y vector in another image
5) Done
6) Wrote SOR and Richardson
7) Done