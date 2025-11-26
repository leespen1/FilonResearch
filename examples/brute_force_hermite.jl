using Polynomials

"""
The polynomial given for ℓ_10 for s=3 in the highly oscillatory text seems
incorrect. I will assume the factor of (1+x)^4 is correct, and do a brute-force
search for the coeffecients of the remaining polynomial factor.
"""
function find_hermite_ℓ10_s3()
    one_plus_x_factor = Polynomial((1,1))^4
    for I in CartesianIndices((-50:50, -50:50, -50:50, -50:50))
        c1, c2, c3, c4 = I.I
        poly = one_plus_x_factor * Polynomial((c1,c2,c3,c4))
        if (derivative(poly, 0)(1) != 0 &&
            derivative(poly, 1)(1) == 0 &&
            derivative(poly, 2)(1) == 0 &&
            derivative(poly, 3)(1) == 0)

            println("Success with ($c1, $c2, $c3, $c4)")

        end
    end
end

function find_hermite_ℓ12_s3()
    one_plus_x_factor = Polynomial((1,1))^4
    for I in CartesianIndices((-50:50, -50:50, -50:50, -50:50))
        c1, c2, c3, c4 = I.I
        poly = one_plus_x_factor * Polynomial((c1,c2,c3,c4))
        if (derivative(poly, 0)(1) == 0 &&
            derivative(poly, 1)(1) == 0 &&
            derivative(poly, 2)(1) != 0 &&
            derivative(poly, 3)(1) == 0)

            println("Success with ($c1, $c2, $c3, $c4)")

        end
    end
end

