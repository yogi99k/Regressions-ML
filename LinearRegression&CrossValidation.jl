### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ 01a47672-c066-4b2f-a4e8-b9fac1481898
begin
	using LinearAlgebra
	using Random
	using DelimitedFiles
end

# ╔═╡ 4b30b45d-f2d2-4245-9516-9b75b8417045
md"""

# **Homework 1** - _Linear Regression, Cross Validation, and Nonlinear Regression_
`EE/CS5841`, Spring 2024


"""

# ╔═╡ 21ec3dc6-b31d-11ee-3008-d7cac2cc072f
md"""

# Part 1: Linear Regression by least-squares (10 points)

"""

# ╔═╡ 18d82c87-0fd8-4b13-b4e7-e80bfb646a41
md"""
### Function definitions:

#### linear\_regression\_train

Creates weights for a least-squares solution for linear regression.

Inputs: 

x - n samples x m features

y - n samples x 1 (for now we can assume that we have only a single output)

Outputs:

w - m features x 1 (again, assuming single output variable)

#### linear\_regression\_infer

Predicts an output based on weights and inputs

Inputs:

x - n samples x m features

w - m features x 1

Outputs
y_predicted - n samples x 1

### Hints:
- The LinearAlgebra library includes useful things like pseudoinverse
- We can transpose with transpose() or using '
- We can check dimensions of matrices with size()
- Some useful matrices can be made with ones(), zeros(), and eye()
- Julia expects us to specify type for the above matrices, just use Float64 for most things unless you want an Int
- We can force arrays to have singleton dimensions with a ;; - see the simple test below for an example
	"""


# ╔═╡ 89eee90a-2227-4167-8ded-cf4be9ac31cd
function linear_regression_train(x,y)
	# your code here!
	#used internet and gpt for reference
	#ones_column = ones(size(x,1))
	#A = hcat(ones_column, x)
	x=hcat(ones(size(x,1)),x)
	weights = pinv(transpose(x)*x)*transpose(x)*y

    # Compute the weights using least squares solution
    #weights = pinv(A' * A) * A' * y
	#println(A)
	return weights
end

# ╔═╡ 03f30763-bd3f-49ad-9066-9b2bbfe4846d
function linear_regression_infer(x, w)
	#refered from internet
	x=hcat(ones(size(x,1)),x)
	
    predicted_y = x * w
    
    return predicted_y
end


# ╔═╡ 70f3150e-b0e7-49a7-b863-39403ee29507
md"""
Here's a simple test
"""


# ╔═╡ d5718e88-8ea7-4589-88e5-bcd7a707b639
begin #we have to use these for Pluto notebooks when we execute multiple things in the same cell
	
	x_demo = [1.0; 2.0; 3.0 ;;]
	y_demo = [3.0; 5.0; 7.0 ;;] # the ";;" makes it a column vector

	try
		w = linear_regression_train(x_demo,y_demo)
		y_demo_pred = linear_regression_infer(x_demo,w)
		print(y_demo_pred)
		if norm(y_demo - y_demo_pred) < 1e-10
			print("It works!")
		else
			print("Not working yet")
		end
		
	catch 
		print("Something's wrong, perhaps dimensions aren't matching in your function?")
	end

end


# ╔═╡ ed3afd58-7229-44b2-a50a-dcead48aadd9
md"""
# Part 2, test train splitting and cross validation (10 points)
"""

# ╔═╡ ae7f1b33-b5d1-4187-9830-ee48ca1a6084
md"""
### Function definitions:

#### mse

Mean squared error

Inputs:

y - a vector of true values
y_est - a vector of estimated values

Outputs:

error - the mean squared error between the two vectors

#### test\_train\_split

Generates some indices for randomly splitting data. Assume an 80/20 split can just be hardcoded in. It is also acceptable to have this function take in the data instead of the number of points and the first version of the assignment did this.

Inputs:

n - number of data points

Outputs:

test\_train\_indices - a vector of 1's and 0's where 1's represent the entries to be used for test and the 0's represent entries to be used for training

#### cross\_validate\_split

Generate some indices for a cross validation

Inputs:

n - number of data points

k - number of splits for cross validation

Outputs:

split\_indices - a vector of values representing which validation split that entry belongs to


####  eval_model

Function that allows you to pass in data, test/train split indices, model and loss functions, and runs everything for you

Inputs:

x - n samples x m features

y - n samples x 1 (for now we can assume that we have only a single output)

test\_train\_indices - a vector of 1's and 0's where 1's represent the entries to be used for test and the 0's represent entries to be used for training

model\_train - name of the function that you are using to create the model, assumes it uses the same model representation as model_infer

model\_infer - name of the function you are using to run inference on the model

loss - function used to compute loss

Outputs: 

error - the error for the 
#### run\_cross\_validation

Function that allows you to pass in data, cross validation split indices, model and loss functions, and runs everything for you

Inputs:

x - n samples x m features

y - n samples x 1 (for now we can assume that we have only a single output)

split\_indices - a vector of 1's and 0's where 1's represent the entries to be used for test and the 0's represent entries to be used for training

model\_train - name of the function that you are using to create the model, assumes it uses the same model representation as model_infer

model\_infer - name of the function you are using to run inference on the model

loss - function used to compute loss

Outputs:

loss\_per\_run - a vector that is the same length as the number of cross validation folds that has the validation loss for that fold. if you're really motivated, you could make it output a training and validation loss seprately

### Hints:
- use vectorized dot operations, for example if we wanted to find everything equal to 1 in an array, we could use '.== 1' and it would return a 'true' for every element that had a 1
	"""


# ╔═╡ 219df1a5-01ce-4bbc-8e5b-4cd245ea9bcd

function mse(y, y_est)
	#your code here 
	#used internet and gpt for reference
	n = length(y)
    if n != length(y_est)
        error("Length of y and y_est must be the same")
    end
    
    error_sum = 0.0
    for i in 1:n
        error_sum += (y[i] - y_est[i])^2
    end
    error = error_sum/n
    return error
    
end



# ╔═╡ 6b76c9c8-f6c6-44ac-a3d5-cbfe4f15ff7b
md"""
Let's write a simple test for this
"""

# ╔═╡ 515d9602-750f-46a9-8d4f-f0176ecd7279
begin
	y1 = [1 0 0]
	y2 = [1 2 0]
	try
		error = mse(y1,y2)
		if error - 1.33333333 < 1e-3
			print("Yay, it works!")
		else 
			print("Something isn't right")
		end
	catch
		print("Something isn't working")
	end
end

# ╔═╡ b558ef69-29d5-4897-9b7b-25f6355d16b2
function test_train_split(n)
    # Generate random values
	#used internet and gpt for reference
    rand_values = rand(n)

    # Convert to binary vector based on 80/20 split
    test_train_indices = rand_values .> 0.8
	print(test_train_indices)

    return test_train_indices
end

# ╔═╡ 510f5bd3-418a-4415-abea-5f44da5e37a4
#lazy visual test, needs asserts/condition checking for a proper test
#we can select from data using this one-hot by making it into a boolean
begin
	try
			
		example_split = test_train_split(10)
		selection_test_x = [1:10 ;;] #force it to be an nx1 if we only have 1 feature
		example_train_indices = iszero.(example_split)
		test_indices = .!example_train_indices
		selection_test_x[test_indices,:]
	catch
	end
end

# ╔═╡ 53bfd80a-d089-45fc-8edc-5a7f84efe386
function cross_validate_split(n,k)
	#used internet and gpt for reference
	# Your code here
	fold_size = div(n, k)
    split_indices = zeros(Int, n)

    for i in 1:k
        start_idx = (i - 1) * fold_size + 1
        end_idx = min(i * fold_size, n)
        split_indices[start_idx:end_idx] .= i
    end

	
	
	return split_indices
end

# ╔═╡ a8919408-d0ea-4038-9df5-603ba978ed1b
#little visual test - it will update once the above code works
try
	cross_validate_split(10,5)
catch
end


# ╔═╡ 9f97dc53-feaa-40b6-aca9-150385a3caf5
#some demo code to show you how you could use the split indices to separate some data
#this may be useful a little bit further down...
begin
	try
			
		example_split = cross_validate_split(10,5)
		selection_test_x = [1:10 ;;] #force it to be an nx1 if we only have 1 feature
		splits = zeros(5,2)
		for split = 1:5
			val_set_indices = findall(==(split),example_split)
			splits[split,:] = selection_test_x[val_set_indices,:]
		end
		print(splits)
	catch
	end
end

# ╔═╡ edcbb313-7abb-4667-a423-49418ad8552f
md"""
This next part is just combining stuff from Part 1 with the loss function (MSE) from Part 2. It's just to see how we can pass functions into functions to do things for us.
"""

# ╔═╡ f2c154b1-00fd-4893-af4c-ce674defe81b
function eval_model(x,y,test_train_indices,model_train,model_infer,loss)
	#used internet and gpt for reference
	x = [x ;;] # force it to be an nX1 instead of just an n-element array if x is 1D
	#if ndims(x) == 1
        #x = reshape(x, :, 1)
    #end
    
    # Split the data into training and testing sets based on test_train_indices
	#used internet and gpt for reference
    x_train = x[test_train_indices .== 0, :]
    y_train = y[test_train_indices .== 0, :]
    x_test = x[test_train_indices .== 1, :]
    y_test = y[test_train_indices .== 1, :]

	x_train = [x_train ;;]
	x_test = [x_test ;;]
	y_train = [y_train ;;]
	y_test = [y_test ;;]

    # Train the model
    model = model_train(x_train, y_train)

    # Infer predictions for the test set
    y_pred = model_infer(x_test, model)

    # Calculate the loss
    error = loss(y_test, y_pred)

    #return error
	

    
	return error
end

# ╔═╡ 82e0291f-2eba-4eda-9331-95ba4497c1e8
md"""
Let's reuse our test code from Part 1, modified slightly. This isn't a thorough test, as it's a zero-error case with almost no data, but we could write a second test with a higher known error if we wanted to include that as well.
"""

# ╔═╡ 644953b5-8fd4-4b88-8033-1a0522dbe748
begin 
	try
		test_train_indices = [0,1,0]
		loss = eval_model(x_demo,y_demo,test_train_indices,linear_regression_train,linear_regression_infer,mse)
	
		if loss < 1e-3
			print("It works!")
		else
			print("Not working yet")
		end
		
	catch 
		print("Something's wrong, perhaps dimensions aren't matching in your function?")
	end

end

# ╔═╡ 2bc91610-f99a-42e7-9f07-770f883090dc
function run_cross_validation(x, y, split_indices, model_train, model_infer, loss)
	#used internet and gpt for reference
    x = [x ;;] #force it to be an nx1 if we only have 1 feature
    y = [y ;;]

    num_folds = maximum(split_indices)
    loss_per_run = zeros(num_folds)

    for i in 1:num_folds
        x_train = x[split_indices .!= i, :]
        y_train = y[split_indices .!= i, :]

        x_test = x[split_indices .== i, :]
        y_test = y[split_indices .== i, :]

        # Train the model
        model = model_train(x_train, y_train)

        # Infer predictions for the validation set
        y_pred = model_infer(x_test, model)

        # Compute loss for this fold
        error = loss(y_test, y_pred)
        loss_per_run[i] = error
    end

    return loss_per_run
end


# ╔═╡ c1d4fd6e-76fc-4c8a-8b5c-0344ab388d0e
md"""

Test it out on some synthetic data. 
Create a line with a slope of 2 and an intercept of 1 and add in some Gaussian noise

Run 5-fold cross validation on it and don't worry about a test/train split for now.
"""

# ╔═╡ 29c24896-4e7c-42b7-ad9c-3ad2a9856b06
begin
	try
		x = collect(1:100)
		y = x*2+ones(Float64,100)+randn(Float64,100)
		split_indices = cross_validate_split(100,5)
		loss_per_run = run_cross_validation(x,y,split_indices,linear_regression_train,linear_regression_infer,mse)
		mean_error = sum(loss_per_run)/5
		#println("Loss per run: ", loss_per_run)
        #println("Mean error: ", mean_error)
	catch e
		 println("Error occurred: ", e)
	end
end

# ╔═╡ 1d402830-b6db-42f2-a600-604eceb38928
md"""
# Part 3 Regularized Linear Regression (10 points)
For now we will just use L2 regularization, so modify your code from the first set of linear regression functions to include an L2 regularization term

#### regularized\_linear\_regression\_train
Performs L2-regularized linear regresion

Inputs:

x - n samples x m features

y - n samples x 1 (for now we can assume that we have only a single output)

lambda - regularization constant

Outputs:

w - m features x 1 (again, assuming single output variable)



"""

# ╔═╡ ca624b79-ca39-4a4e-a437-864a2df985f5
function regularized_linear_regression_train(x, y, l = 0.1)
	#used internet and gpt for reference
	x = hcat(ones(size(x, 1)), x)
	weights = pinv(transpose(x) * x + l * I) *transpose(x) * y
	return weights
end

# ╔═╡ c062bb1f-b5ba-46df-9fb0-e22f45ef235f
#let's do a simple test
try
	begin 
		x_demo_reg = 1.0*[1 1; 3 3; 4 4]
		y_demo_reg = [3.0; 5.0; 7.0 ;;] # the ";;" makes it a column vector
		w_reg01 = regularized_linear_regression_train(x_demo_reg,y_demo_reg,0.1)
	end
catch
end

# ╔═╡ 655752f5-eef9-4f3d-bbf1-b6a71ad3d8c8
#for fun we can vary the regularization constant
try
	begin 
		x_demo_reg = 1.0*[1 1; 3 3; 4 4]
		y_demo_reg = [3.0; 5.0; 7.0 ;;] 
		w_reg1 = regularized_linear_regression_train(x_demo_reg,y_demo_reg,1)
	end
catch
end
	

# ╔═╡ 1d33f6cc-87ea-4cc4-a6ca-c5fc40ff242d
#we can compare with no regularization as well
try
	begin 
		x_demo_reg = 1.0*[1 1; 3 3; 4 4]
		y_demo_reg = [3.0; 5.0; 7.0 ;;] 
		w_noreg = linear_regression_train(x_demo_reg,y_demo_reg)
	end
catch e
	print("wrong")
end
	

# ╔═╡ 8c81340d-ca38-4616-8d84-fd7598e26f87
md"""
# Part 4 Basis functions (10 points)
Modify the function below to create n polynomials for the input features x in increasing power

#### polynomial_basis
Input: 

x - n samples x m features
n - highest polynomial to create

Output:

xns - n samples x mXn features. features should be ordered as x1n1 x2n1 ... xmn1 x1n2 x2n2 ... xmn2 ... xmnn for each row of samples

### Hints:
- We can concatenate arrays using square brackets
- We can do elementwise operations with the dot operator. ie. x.^2 squares each element
"""

# ╔═╡ 3db218d8-d8a6-4647-8bb1-ee74c575a9aa
function polynomial_basis(x, n)
	#used internet and gpt for reference
    # Initialize an empty matrix to store the polynomial features
	#used internet and gpt for reference
    xns = zeros(size(x, 1), size(x, 2) * n)

    # Create polynomial features for each input variable
    for i in 1:size(x, 2)
        for j in 1:n
            # Compute the j-th power of each element in x[:, i]
            poly_feature = x[:, i] .^ j
            # Assign the computed feature to the appropriate column in xns
            xns[:, (i-1) * n + j] = poly_feature
        end
    end

    return xns
end

# ╔═╡ ed4ac941-ad5f-46e8-9d8b-740ec143b795
begin
	#Download the data and load it using DelimitedFiles function
	#update the code with your file location or use another loading method
	#used internet and gpt for reference
	using Statistics
	begin
	    try
	        data1 = readdlm("C:\\Users\\nikhi\\Downloads\\yacht_data.csv", ',')
	        global data = convert(Array{Float64,2}, data1[:,1:7])
	    catch
	        println("data not process or imported")
	        
	    end
		println(size(data))
	    a = data[:, 1:end-1]
	    b = data[:, end]
	    n = size(a, 1)
	    test_train_indices = test_train_split(n)
	    # Applying regularized linear regression
	    println("Regularized Linear Regression :\n")
	    
	    lambda_values = [0.01, 0.1, 1.0]
	    b_lambda = 0
	    b_cv_loss = Inf
		split = cross_validate_split(n, 5)
	
		
		for l in lambda_values
   			 model_train = (x, y) -> regularized_linear_regression_train(x, y, l) 
    		cv_loss = run_cross_validation(a, b, split, model_train, linear_regression_infer, mse)
    
    		average_cvloss = mean(cv_loss)
    		println("Lambda = $l, CV Loss = $(mean(cv_loss))")
    
    		if average_cvloss < b_cv_loss
        		global b_cv_loss = average_cvloss
        		global b_lambda = l
			end
		end
		println("\nBest Lambda: $b_lambda with CV Loss: $b_cv_loss")
		train_indices = test_train_indices .== 0
		test_indices = test_train_indices .== 1
		x_train = x[train_indices, :]
		y_train = y[train_indices]
		x_test = x[test_indices, :]
		y_test = y[test_indices]

	    #println("\nBest Lambda: $b_lambda with CV Loss: $b_cv_loss")
		b_mod_wts = regularized_linear_regression_train(x_train, y_train, b_lambda)
		y_pred_test = linear_regression_infer(x_test, b_mod_wts)
		test_error = mse(y_test, y_pred_test)
		println("\nError on the hold out dataset: $test_error\n\n")

		
		a_new = Matrix(a)
		deg = [1,20,3,40]
		for d in deg
	    	x_poly = polynomial_basis(a_new, d)
	    	performance_poly = run_cross_validation(x_poly, b, split, regularized_linear_regression_train, linear_regression_infer, mse)
	   		println("Cross-Validated MSE for polynomial degree $d: ", 			          sum(performance_poly) / length(performance_poly))
	    end
end

end

# ╔═╡ 2229208a-674e-4a62-8f62-8375299d408d
#simple visual test
begin
	try
		x_mat = [1 2; 3 4]
		xmat_n2= polynomial_basis(x_mat,3)
		
	catch
	end
end

# ╔═╡ 37d2448b-2a7e-4468-93b0-f36fa721cc64
md"""
# Part 5 Put it all together! (10 points)
Download and load the following dataset:
Yacht Hydrodynamics: https://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics
Load the data and run some experiments with it using regularized linear or polynomial regression. 

For full credit, separate off 20% as a test set, explore at least 3 options (regularization and/or polynomial basis functions), use cross validation to select the best one, and report the error on the held out dataset. Most importantly, report your conclusions and justify it with a sentence or two!!
"""

# ╔═╡ 611f7617-1d5a-4073-b54c-10136cb45155


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DelimitedFiles = "~1.9.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "b6bd11996360076ea4499c59c8652db519033617"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"
"""

# ╔═╡ Cell order:
# ╟─4b30b45d-f2d2-4245-9516-9b75b8417045
# ╟─21ec3dc6-b31d-11ee-3008-d7cac2cc072f
# ╠═01a47672-c066-4b2f-a4e8-b9fac1481898
# ╟─18d82c87-0fd8-4b13-b4e7-e80bfb646a41
# ╠═89eee90a-2227-4167-8ded-cf4be9ac31cd
# ╠═03f30763-bd3f-49ad-9066-9b2bbfe4846d
# ╟─70f3150e-b0e7-49a7-b863-39403ee29507
# ╠═d5718e88-8ea7-4589-88e5-bcd7a707b639
# ╟─ed3afd58-7229-44b2-a50a-dcead48aadd9
# ╟─ae7f1b33-b5d1-4187-9830-ee48ca1a6084
# ╠═219df1a5-01ce-4bbc-8e5b-4cd245ea9bcd
# ╟─6b76c9c8-f6c6-44ac-a3d5-cbfe4f15ff7b
# ╠═515d9602-750f-46a9-8d4f-f0176ecd7279
# ╠═b558ef69-29d5-4897-9b7b-25f6355d16b2
# ╠═510f5bd3-418a-4415-abea-5f44da5e37a4
# ╠═53bfd80a-d089-45fc-8edc-5a7f84efe386
# ╠═a8919408-d0ea-4038-9df5-603ba978ed1b
# ╠═9f97dc53-feaa-40b6-aca9-150385a3caf5
# ╟─edcbb313-7abb-4667-a423-49418ad8552f
# ╠═f2c154b1-00fd-4893-af4c-ce674defe81b
# ╟─82e0291f-2eba-4eda-9331-95ba4497c1e8
# ╠═644953b5-8fd4-4b88-8033-1a0522dbe748
# ╠═2bc91610-f99a-42e7-9f07-770f883090dc
# ╟─c1d4fd6e-76fc-4c8a-8b5c-0344ab388d0e
# ╠═29c24896-4e7c-42b7-ad9c-3ad2a9856b06
# ╟─1d402830-b6db-42f2-a600-604eceb38928
# ╠═ca624b79-ca39-4a4e-a437-864a2df985f5
# ╠═c062bb1f-b5ba-46df-9fb0-e22f45ef235f
# ╠═655752f5-eef9-4f3d-bbf1-b6a71ad3d8c8
# ╠═1d33f6cc-87ea-4cc4-a6ca-c5fc40ff242d
# ╟─8c81340d-ca38-4616-8d84-fd7598e26f87
# ╠═3db218d8-d8a6-4647-8bb1-ee74c575a9aa
# ╠═2229208a-674e-4a62-8f62-8375299d408d
# ╟─37d2448b-2a7e-4468-93b0-f36fa721cc64
# ╠═ed4ac941-ad5f-46e8-9d8b-740ec143b795
# ╠═611f7617-1d5a-4073-b54c-10136cb45155
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
