import numpy as np

def structured_loss(logits, y, alpha=1.0):
    l2_norm = np.sum(logits * logits, axis = 1, keepdims = True)
    t1 = l2_norm @ np.ones_like(l2_norm).T
    t2 =  np.ones_like(l2_norm) @ l2_norm.T
    t3 = 2.0*(logits @ logits.T)

    pairwise_dist = np.sqrt(t1 + t2 - t3)
    positive_pairs = y @ y.T
    negative_pairs = ~positive_pairs.astype(np.bool)
    negative_pairs = negative_pairs.astype(positive_pairs.dtype)

    print("pos pairs\n", positive_pairs)
    print("neagtive _pairs\n", negative_pairs)
    print("dists\n", pairwise_dist)
    print(alpha - pairwise_dist)

    exp_margin = np.exp(alpha - pairwise_dist)
    print(exp_margin)


    J = exp_margin * negative_pairs + 1e-3
    print("J\n", J)
    
    Jik = np.sum(J, axis = 0, keepdims = True)
    Jjl = np.sum(J, axis = 1, keepdims = True)
    
    print("Jik", Jik.flatten(), "Jjl", Jjl.flatten())
    print("log(sums)\n", np.log(Jik + Jjl))
    

    Jtilde = np.log(Jik + Jjl) + pairwise_dist
    print(Jtilde * positive_pairs)
    loss = np.sum(np.maximum(0, Jtilde * positive_pairs) ** 2)
    loss = 0.5 * (1/np.sum(positive_pairs)) * loss
    print(loss)

def main():
  logits = np.array([[1, 1, 1], [1.2, 1.2, 1.2], [1.2, 1.2, 1.2]])
  y = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
  structured_loss(logits, y, alpha = 1.0)


if __name__ == '__main__':
  main()

