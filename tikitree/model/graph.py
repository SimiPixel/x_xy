ROOT = -1


def check_parent_array(parent):
    "verify equation (2)"
    for i in range(len(parent)):
        assert ROOT <= parent[i] < i


def is_root(i: int) -> bool:
    return ROOT == i


class Graph:
    def __init__(self, parent_array):
        check_parent_array(parent_array)
        self._parent_array = parent_array

    @property
    def N(self):
        return len(self._parent_array)

    def lambda_(self, i: int) -> int:
        """parent of body `i`"""
        return self._parent_array[i]

    def parent(self, i: int) -> int:
        assert not is_root(i)
        return self.lambda_(i)

    def mu_(self, i: int) -> list[int]:
        """ordered set of children of body `i`"""
        children = set()
        for body, parent in enumerate(self._parent_array):
            if parent == i:
                children.add(body)
        return sorted(children)

    def kappa_(self, i: int) -> list[int]:
        """ordered set of joints on path between body `i` and root"""
        joints = set()
        parent = self.lambda_(i)
        while parent != ROOT:
            joints.add(i)
            i = parent
            parent = self.lambda_(i)
        return sorted(joints)

    def nu_(self, i: int) -> list[int]:
        """ordered set of bodies in the subtree starting at body `i`"""
        descendants = set([i])

        children = self.mu_(i)
        # is leaf
        if len(children) == 0:
            return sorted(descendants)

        # else
        for child in children:
            descendants.add(child)
            for childs_descendant in self.nu_(child):
                descendants.add(childs_descendant)

        return sorted(descendants)
