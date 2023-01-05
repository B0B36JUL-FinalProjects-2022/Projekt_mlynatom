using Projekt_mlynatom
using Test

@testset "Projekt_mlynatom.jl" begin

    @testset "utils.jl" begin

        @testset "compute_class_error" begin
            x1 = [1, 0, 0, 1]
            x2 = [1, 1, 0, 0]
            x3 = [0, 1, 1, 0]
            res1 = 0.5
            @test compute_class_error(x1, x2) == res1
            @test compute_class_error(Bool.(x1), x2) == res1
            @test compute_class_error(x1, Bool.(x2)) == res1
            @test compute_class_error(Bool.(x1), Bool.(x2)) == res1
            res2 = 0
            @test compute_class_error(x1, x1) == res2
            res3 = 1
            @test compute_class_error(x1, x3) == res3
        end

        @testset "read_csv_to_df" begin
            @test_throws ErrorException read_csv_to_df("data/asdf.csv")
        end

    end

    @testset "data_preparation.jl" begin
        cat_vec = ["a", "b", "c", "c"]
        one_hot_mat = [1 0 0; 0 1 0; 0 0 1; 0 0 1]
        @testset "categorical_to_one_hot" begin

            @test categorical_to_one_hot(cat_vec) == one_hot_mat
        end

        @testset "categorical_to_dummy_encoding" begin
            dummy_mat = [0 0; 1 0; 0 1; 0 1]
            @test categorical_to_dummy_encoding(cat_vec) == dummy_mat
        end

        @testset "one_hot_to_one_cold" begin
            one_cold_vec = [0, 1, 2, 2]
            @test one_hot_to_one_cold(one_hot_mat; dims=1) == one_cold_vec
        end

        @testset "standardize" begin
            test_mat = [-1 -1; 0 0; 1 1]
            @test standardize(test_mat; dims=1) == test_mat
            @test standardize(test_mat, test_mat; dims=1) == (test_mat, test_mat)
            @test standardize(test_mat'; dims=2) == test_mat'
            @test standardize(test_mat', test_mat'; dims=2) == (test_mat', test_mat')
        end

        @testset "split_dataset" begin
            test_mat = ones((10,2))
            test_y = ones(10)

            @test split_dataset(test_mat, test_y; dev_ratio=0.1) == (ones((9,2)), ones(9), ones((1,2)), ones(1))
        end

        @testset "count_all" begin
            vec = ["a", "b", "c", "c", "c", "a"]
            ret_dict = Dict([(:a, 2), (:b, 1), (:c, 3)])
            @test count_all(vec) == ret_dict
        end


    end

    @testset "nn.jl" begin
        
    end

    @testset "log_reg.jl" begin
        
    end

end
