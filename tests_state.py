from state import State


s = State(10, [])

print("Not empty ", s.is_empty())
if 5 not in s:
    s.add(5)

print("cardinality ", s.cardinality())
print("to list ", s.to_list())
print("No more empty ", not s.is_empty())
print("But not full ", not s.is_full())
for i in range(11):
    s.add(i)

print("Now full ", s.is_full())

s2 = State.added(s, 10)
print("Copy of it ", s2.to_list())
