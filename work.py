from predict import SchoolClosureAgent

test_case = 0
location_validator = 'unspecific'
test_inputs = 'Will school be closed tomorrow in my town'

response = SchoolClosureAgent.get_valid_location(self,  location_validator, test_inputs)
print(response)