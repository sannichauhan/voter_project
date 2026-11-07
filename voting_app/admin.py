from django.contrib import admin
from .models import VoterData

@admin.register(VoterData)
class VoterAdmin(admin.ModelAdmin):
    list_display = ('serial_no', 'voter_id', 'name', 'relation_name', 'house_no', 'age', 'gender')
    search_fields = ('name', 'voter_id', 'house_no')