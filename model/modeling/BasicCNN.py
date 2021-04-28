'''
-------------------------------
Network Architecture
1.Convolution, Filter shape:(5,5,6), Stride=1, Padding=’SAME’
2. Max pooling (2x2), Window shape:(2,2), Stride=2, Padding=’Same’
3. ReLU
4. Convolution, Filter shape:(5,5,16), Stride=1, Padding=’SAME’
5. Max pooling (2x2), Window shape:(2,2), Stride=2, Padding=’Same’
6. ReLU
7. Fully Connected Layer (128)
8. ReLU
9. Fully Connected Layer (10)
10. Softmax
------------------------------------------
'''

import ../