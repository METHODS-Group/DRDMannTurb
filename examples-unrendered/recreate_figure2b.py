####
# Figure 2b recreation
#
# \sigma^2 = 2 (m/s)^2
# z_i = 500 m
# psi = 45 degrees
#
# Grid dimensions: L_2D x 0.125 L_2D

sigma2 = 2
z_i = 500.0
psi_deg = 45.0


# def analytic_F11_2D(k1):
#     """
#     Analytic solution for F11(k1) in 2d.
#     """
#     a = 1 + 2 * (k1 * generator.L_2D * np.cos(generator.psi_rad))**2
#     b = 1 + 2 * (k1 * generator.z_i * np.cos(generator.psi_rad))**2
#     p = (L_2D**2 * b) / (z_i**2 * a)

#     d = 1.0

#     first_term_numerator = (sp.gamma(11/6) * (L_2D**(11/3)))\
#         * (-p * sp.hyp2f1(5/6, 1, 1/2, p) - 7 * sp.hyp2f1(5/6)\
#            + 2 * sp.hyp2f1(-1/6, 1, 1/2, p))
#     first_term_denominator = (2 * np.pi)**(11/3) * (a**(11/6))

#     second_term_numerator = (L_2D**(14/3) * np.sqrt(b))
#     second_term_denominator = (2 * np.sqrt(2) * d**(7/3) * (generator.z_i * np.sin(generator.psi_rad)))

#     return generator.c * ((first_term_numerator / first_term_denominator) \
#                           + (second_term_numerator / second_term_denominator))


# def analytic_F22_2D(k1):

#     a = 1 + 2 * (k1 * generator.L_2D * np.cos(generator.psi_rad))**2
#     b = 1 + 2 * (k1 * generator.z_i * np.cos(generator.psi_rad))**2
#     p = (L_2D**2 * b) / (z_i**2 * a)

#     d = 1.0

#     leading_factor_num = generator.z_i**4 * a**(1/6) ** generator.L_2D * sp.gamma(17/6)
#     leading_factor_denom = 55 * np.sqrt(2 * np.pi) * (generator.L_2D**2 - generator.z_i**2)**2\
#         * b * sp.gamma(7/3) * np.sin(generator.psi_rad)
#     leading_factor = - leading_factor_num / leading_factor_denom

#     line_1 = -9 - 25 * sp.hyp2f1(-1/6, 1, 1/2, p)
#     line_2 = p**2 * ( 15 - 30 * sp.hyp2f1(-1/6, 1, 1/2, p) - 59 * sp.hyp2f1(5/6, 1, 1/2, p))
#     line_3 = 35 * sp.hyp2f1(5/6, 1, 1/2, p) + 15 * p**3 * sp.hyp2f1(5/6, 1, 1/2, p)
#     line_4 = p * (-54 + 88 * sp.hyp2f1(-1/6, 1, 1/2, p) + 9 * sp.hyp2f1(5/6, 1, 1/2, p))

#     term_1 = leading_factor * (line_1 + line_2 + line_3 + line_4)

#     term_2 = (L_2D**(14/3)) / (np.sqrt(2 * b) * d**(7/3) * z_i * np.sin(generator.psi_rad))

#     paren = term_1 - term_2

#     return generator.c * k1**2 * paren
