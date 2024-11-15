def tendon_length_pin_joint(theta, R1):
   """Computes tendon length for a given theta
   """
   return -R1 * theta


def get_tendon_lengths_lambda(theta_PIP, theta_MCP, theta_ABD, muscle_group):
   tendon_lengths = []
   radius_mm = muscle_group.joint_radius

   if muscle_group.name == "thumb":

      # PIP flexion
      tl0 = tendon_length_pin_joint(theta_PIP, radius_mm[0][0])

      # MCP flexion (thumb)
      tl1 = tendon_length_pin_joint(theta_MCP, radius_mm[1][0])

      # Abduction (thuumb)
      tl2 = tendon_length_pin_joint(theta_ABD, radius_mm[2][0])

      # Each joint is pretty much independent
      tendon_lengths = [tl0, tl1, tl2]
   else:
      # PIP flexion
      tl0 = tendon_length_pin_joint(theta_PIP, radius_mm[0][0])

      # MCP flexion
      tl1 = tendon_length_pin_joint(theta_MCP, radius_mm[1][0])

      # Abduction 
      tl2 = tendon_length_pin_joint(theta_ABD, radius_mm[2][0])

      tendon_lengths = [tl0, tl1, tl2]

   return tendon_lengths
