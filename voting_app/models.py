from django.db import models


class VoterData(models.Model):
    serial_no = models.CharField(max_length=10, blank=True, default="")
    voter_id = models.CharField(
        max_length=50,
        null=True,
        blank=True
    )
    name = models.CharField(max_length=100, blank=True, default="")
    relation_name = models.CharField(max_length=100, blank=True, default="")
    house_no = models.CharField(max_length=50, blank=True, default="")
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=20, blank=True, default="")

    def __str__(self):
        return f"{self.voter_id or 'ID Pending'} - {self.name}"