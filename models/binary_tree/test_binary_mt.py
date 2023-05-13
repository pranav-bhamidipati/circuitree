import numpy as np
from binary_tree import BinaryTree

if __name__ == "__main__":
    depth = 6
    n_leaves = 2**depth

    bt = BinaryTree(depth=depth)

    print(bt.modularity)
    ...

    print("Now testing with a random tree")

    size = 25
    results = list(
        map(lambda _: BinaryTree(depth=depth, random=True).modularity, range(size))
    )
    # for _ in range(100):
    #     bt = BinaryTree(depth=depth, random=True)
    #     print(bt.modularity)

    print(np.array(results))

    ...

    print("Now computing the modularity of some pre-specified trees")
    print("(1) A tree with 2^depth outcomes, all of which are successes")
    print(
        BinaryTree(
            depth=depth, success_codes=[f"{i:03b}" for i in range(n_leaves)]
        ).modularity
    )

    print("(2) A tree with 2^depth outcomes, none of which are successes")
    print(BinaryTree(depth=depth, success_codes=[]).modularity)

    print("(3) Trees with 2^depth outcomes, half of which are successes")
    sc1 = [f"{i:0{depth}b}" for i in range(1, n_leaves, 2)]
    sc2 = [f"{i:0{depth}b}" for i in range(n_leaves // 4, 3 * n_leaves // 4)]
    sc3 = [f"{i:0{depth}b}" for i in range(n_leaves // 2, n_leaves)]
    mt1 = BinaryTree(depth=depth, success_codes=sc1).modularity
    mt2 = BinaryTree(depth=depth, success_codes=sc2).modularity
    mt3 = BinaryTree(depth=depth, success_codes=sc3).modularity

    print("----- Every other outcome is a success")
    print("\t", mt1)
    print("----- The middle half of outcomes are successes")
    print("\t", mt2)
    print("----- The last half of outcomes are successes")
    print("\t", mt3)

    ...
