from operator import itemgetter


def lowercase(func):
    def wrapper():
        resp = func()
        return resp.lower()
    return wrapper


def splitter(func):
    def wrapper():
        resp = func()
        return resp.split()
    return wrapper


@splitter
@lowercase
def hello():
    return "Hello World"


if __name__ == '__main__':
    # print(hello())
    d = [{'name': 'Pune', 'pop': 897}, {'name': 'Kolkata', 'pop': 897}, {'name': 'Pune', 'pop': 1000}]
    keys = list(d[0].keys())
    sorted_tuples = map(lambda y: {keys[0]: y[0], keys[1]: y[1]}, sorted(map(lambda x: (x['name'], x['pop']), d), key=itemgetter(0, 1)))
    print(list(sorted_tuples))
