from state import State


s = State(10, [])
s_false_copy = s

print("Not empty ", s.is_empty())
if 5 not in s:
    s.add(5)

print("cardinality ", s.cardinality())
print("to list ", s.to_list())
print("No more empty ", not s.is_empty())
print("But not full ", not s.is_full())
for i in range(10):
    s.add(i)

print("Now full ", s.is_full())

s2 = State.added(s, 9)
print("Copy of it ", s2.to_list())
print("False copy of it at start ", s_false_copy.to_list())
