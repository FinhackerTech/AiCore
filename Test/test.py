import json
jsonList =  [1,2,3]
ok = json.dumps(jsonList, ensure_ascii=False)

print(json.loads(ok))