
from graphviz import Digraph

# trace back the operations of the nn
def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

# draw the graph of operations 
def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={'rankdir' : 'LR'})

    # get the nodes
    nodes, edges = trace(root)
    for n in nodes:
        # id return the id of the object *unique
        uid = str(id(n))
        # create a node
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')

        if n._op:
            #so we can dis[lay the operation of the nodes
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    # iterate edges and connect them
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

