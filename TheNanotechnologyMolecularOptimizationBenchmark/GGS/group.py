from group_selfies import Group as selfies_group


class Group(selfies_group):
    def __init__(self, name, canonsmiles, overload_index=None, all_attachment=False, smarts_override=None, priority=0, sanitize=False):
        super().__init__(name, canonsmiles, overload_index, all_attachment, smarts_override, priority, sanitize)
        self._attachment_point_start = -1
        self._reserved_attachment_point = -1
        self._index_in_mol = -1

        self.attachment_point_occupation = [0] * len(self.attachment_points)
        self.occupied_attachment_points = []


    @property
    def attachment_point_start(self):
        return self._attachment_point_start

    @attachment_point_start.setter
    def attachment_point_start(self, value):
        if self._attachment_point_start == -1:
            if value < 0 or value >= len(self.attachment_points):
                raise ValueError("Invalid attachment_point_start point")
            self._attachment_point_start = value

    @property
    def reserved_attachment_point(self):
        return self._reserved_attachment_point

    @reserved_attachment_point.setter
    def reserved_attachment_point(self, value):
        self._reserved_attachment_point = value
        
    @property
    def index_in_mol(self):
        return self._index_in_mol
    
    @index_in_mol.setter
    def index_in_mol(self, value):
        if self._index_in_mol != -1:
                raise ValueError(f"Index in mol has already been set. Cannot set it again for {self.name}.")
        self._index_in_mol = value

    def reset_attachment_point_start(self):
        """
        Reset the attachment point start to -1 -> needed for anchor pos mutation
        :return:
        """
        self._attachment_point_start = -1


            

    def occupy_attachment_point(self, point):
        """
        Occupy the attachment point. Return True if successful, False if not.
        :param point:
        :return:
        """
        self.attachment_point_start = point
        if point < 0 or point >= len(self.attachment_points):
            raise ValueError(f"Invalid attachment point {point}, {self.attachment_point_occupation}, {self.reserved_attachment_point}")
        if self.attachment_point_occupation[point] == 0:
            self.attachment_point_occupation[point] = 1
            self.occupied_attachment_points.append(point)
            if(point == self.reserved_attachment_point):
                self.reserved_attachment_point = -1

            return True
        else:
            return False

    def free_last_attachment_point(self):
        """
        Free the last attachment point. Return True if successful, False if not.
        :param point:
        :return:
        """
        if len(self.occupied_attachment_points) == 0:
            return False
        last_point = self.occupied_attachment_points[-1]
        return self.free_attachment_point(last_point)
        
    def free_attachment_point(self, point):
        """
        Free the attachment point. Return True if successful, False if not.
        :param point: 
        :return: 
        """
        if self.attachment_point_start == point:
            #start attachment point can never be cleared
            return False
        if point in self.occupied_attachment_points:
            self.attachment_point_occupation[point] = 0
            self.occupied_attachment_points.remove(point)
            return True
        else:
            return False

    def get_free_attachment_points(self):
        """
        Return a list of the free attachment points
        :return:
        """
        free = [i for i in range(len(self.attachment_points)) if self.attachment_point_occupation[i] == 0]
        return free

    def next_avail_attachment(self, shift: int):
        """
        Return the next available attachment point, if it exists, otherwise return -1
        :param shift: 
        :return: 
        """
        if (len(self.attachment_points) - len(self.occupied_attachment_points) < 1):
            return -1
        if len(self.occupied_attachment_points) == 0:
            last_point = 0
        else:
            last_point = self.occupied_attachment_points[-1]
        next_point = (last_point + shift) % len(self.attachment_points)
        while next_point in self.occupied_attachment_points:
            next_point = ( next_point + 1 ) % len(self.attachment_points)
        return next_point
    
    def connect_to_last_attachment_possible(self):
        """
        Return True if it is possible to connect to the last attachment point, False if not
        :return: 
        """
        # fixme: does this cover all cases?
        if len(self.occupied_attachment_points) == 1 and self.occupied_attachment_points[0] == self.attachment_point_start:
            return False
        else:
            return True

    def __str__(self):
        return f"{self.name} {len(self.attachment_points)}"

    def attachment_valency__(self, global_idx):
        """
        Return the valency of the attachment point
        :param global_idx:
        :return:
        """
        print("This is me")
        return 1

    def reserve_attachment_point(self, point):
        """
        Reserve the attachment point. Return True if successful, False if not.
        :param point:
        :return:
        """
        if point < 0 or point >= len(self.attachment_points):
            raise ValueError("Invalid attachment point")
        if self.attachment_point_start == -1:
            raise ValueError(f"Start attachment point not set. Cannot reserve attachment point {point}.")
        #point must be free and not the start attachment point
        if self.attachment_point_occupation[point] == 0 and point != self.attachment_point_start:
            self.reserved_attachment_point = point
            return True
        else:
            return False


if __name__ == '__main__':
    # Test the fragment class

    g2 = selfies_group('pyrazole', 'N1C=CC=N1', all_attachment=True)
    print(g2)
    fragment = Group('toluene', 'CC1=CC=CC=C1', all_attachment=True)
    #fragment = Group('C', 'C', all_attachment=True)
    print(fragment)