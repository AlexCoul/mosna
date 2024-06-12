run_HMRF <- function(r_array, r_df, col_id=1) {
        library(Giotto)
        library(GiottoData)

        max_id = apply( r_array , col_id , function(x) which( x == max(x) , arr.ind = TRUE ) )
        sum_df = sum(r_df)

        detach("package:Giotto", unload=TRUE)
        detach("package:GiottoData", unload=TRUE)

        list(max_id=max_id, sum_df=sum_df)
    }

make_Giottio <- function(coords, markers){ # , network) {
        library(Giotto)
        library(GiottoData)

        instrs = createGiottoInstructions(
            save_plot = FALSE,
            show_plot = FALSE,
            save_dir = './temp/',
            python_path = NULL)

        print(paste('ncol(coords):', ncol(coords)))
        print(paste('seq_len(ncol(coords)):', seq_len(ncol(coords))))


        print(paste('coords:', coords))

        # giotto = createGiottoObject(
        #     expression = markers,
        #     expression_matrix_class = "custom",
        #     expression_feat = "protein",
        #     spatial_locs = coords,
        #     instructions = instrs,
        #     offset_file = NULL)

        detach("package:Giotto", unload=TRUE)
        detach("package:GiottoData", unload=TRUE)

        giotto
    }