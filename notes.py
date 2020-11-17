# Notes on the debias function.
##### NOTE THIS FUNCTION IS TAKEN FROM THE PAPER'S GITHUB REPO.


def debias(E, gender_specific_words, definitional, equalize):
    gender_direction = we.doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)

    # So the 'drop' function drops the gender direction from
    # a specified list of words. I think I got that.
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()

    # This equality part balances pairs across the origin of the gender subspace
    # I think? Thus it is hardly exhaustive. This is particularly important because
    # it seems that exhaustively transforming the entire corpus is not possible
    # since we rely on labeled data. This bodes well for my motivation anyway.
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()