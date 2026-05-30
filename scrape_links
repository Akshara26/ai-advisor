import requests
import time

urls = [
    'https://cse.umn.edu/cs/faculty-instructor-directory',
    'http://www.cs.umn.edu',
    'https://csgrad.appointments.umn.edu',
    'https://policy.umn.edu/education',
    'https://faculty-roles.umn.edu/',
    'https://onestop.umn.edu/academics/degree-completion-steps',
    'https://onestop.umn.edu/academics/examination-committees',
    'https://onestop.umn.edu/academics/grad-and-professional/degree-completion-steps',
    'https://cse.umn.edu/cs/ms-overview#ProjectCoursework',
    'http://www.unite.umn.edu/',
    'https://onestop.umn.edu/academics/submit-your-gpas-approval',
    'https://onestop.umn.edu/sites/onestop.umn.edu/files/forms/um1779_masters_time_extension_request_form.pdf',
    'https://cse.umn.edu/cs/integrated-program-bachelorsmasters',
    'https://www.cs.umn.edu/academics/graduate/phd/transfer-credits',
    'https://onestop.umn.edu/sites/onestop.umn.edu/files/forms/otr028_transfer_credits_undergraduate_to_graduate.pdf',
    'https://onestop.umn.edu/sites/onestop.umn.edu/files/forms/um1777_doctoral_time_extension_request_form.pdf',
    'https://faculty-roles.umn.edu/institution/UMNTC/programs/019660207/responsibilities',
    'https://onestop.umn.edu/academics/doctoral-oral-exam-scheduling',
    'https://docs.google.com/a/umn.edu/forms/d/e/1FAIpQLSeDhNhmgQkktVAsyr8NBi4zWmzEBMieFUEOy6ujBrGKUbK9pg/viewform',
    'https://assets.asr.umn.edu/files/gssp/otr204_Doctor_Phil_Edu_GDP.pdf',
    'https://cei.umn.edu/services-graduate-students-and-postdocs',
    'https://www.cs.umn.edu/academics/undergraduate/engagement/ta_positions',
    'https://hr.umn.edu/Jobs/Student-Job-Center/Graduate-Assistant-Employment/About-Graduate-Assistant-Employment',
    'https://onestop.umn.edu/finances/costs/tuition',
    'https://onestop.umn.edu/finances/tuition',
    'https://www.cs.umn.edu/academics/graduate/ta_handbook?page=ta',
    'https://humanresources.umn.edu/find-job/graduate-assistant-jobs',
    'https://cse.umn.edu/r/career-center/',
    'https://grad.umn.edu/funding',
    'https://grad.umn.edu/funding/current-students/apply-through-program/doctoral-dissertation-fellowship',
    'http://policy.umn.edu/education/gradstudentleave',
    'https://grad.umn.edu/admissions/readmission',
    'https://onestop.umn.edu/academics/registration-times',
    'https://onestop.umn.edu/academics/special-registration-categories-graduate-and-professional-students#accord-4',
    'https://onestop.umn.edu/sites/onestop.umn.edu/files/forms/otr194_application_for_advanced_masters_status.pdf',
    'https://onestop.umn.edu/sites/onestop.umn.edu/files/forms/otr195_application_for_advanced_doctoral_status.pdf',
    'https://cse.umn.edu/cs/graduate-cpt',
    'https://isss.umn.edu/fstudents/employment/cpt',
    'https://isss.umn.edu/fstudents/employment/opt',
    'https://isss.dev.umn.edu/sites/isss.umn.edu/files/documents/cpt-course.pdf',
    'https://onestop.umn.edu/sites/onestop.umn.edu/files/forms/_graduate_student_petition_gdp.pdf',
    'https://isss.umn.edu/fstudent/rcl.html',
    'https://onestop.umn.edu/sites/onestop.umn.edu/files/forms/otr174_umn_medical_supplement.pdf',
    'https://docs.google.com/forms/d/e/1FAIpQLSf_zd8lsW6LLELhVBK5xPAaNUV3hWXv5Y5AoaGCgJA71xGfjw/viewform',
    'https://www.cs.umn.edu/academics/graduate/phd/transfer-credits',
    'https://cse.umn.edu/cseit/classrooms-labs',
    'https://cseit.umn.edu/',
    'https://csgsa.umn.edu/',
    'http://www.grad.umn.edu/admissions/index.html',
    'https://humanresources.umn.edu/find-job/graduateemployment',
    'https://onestop.umn.edu/academics/contact-gssp',
    'https://www.cs.umn.edu/people/faculty',
    'https://www.cs.umn.edu/research/research_areas',
    'https://gsc.umn.edu/',
    'https://gsc.umn.edu/health-care/health-services',
    'https://gsc.umn.edu/programs',
    'https://sass.umn.edu/',
    'https://sass.umn.edu/academic-skills-coaching',
    'https://sass.umn.edu/self-help',
    'https://www.cs.umn.edu/academics/graduate/grad-handbook-archive',
    'https://z.umn.edu/transfercreditstograd',
]

headers = {"User-Agent": "Mozilla/5.0 (compatible; UMN CS Advisor research bot)"}

scrapeable = []
not_scrapeable = []

for url in urls:
    try:
        r = requests.get(url, timeout=8, headers=headers, allow_redirects=True)
        content_type = r.headers.get('content-type', '')
        
        if r.status_code == 200 and 'text/html' in content_type:
            scrapeable.append((r.status_code, url))
        elif r.status_code == 200 and 'pdf' in content_type:
            not_scrapeable.append((r.status_code, 'PDF - skip', url))
        else:
            not_scrapeable.append((r.status_code, content_type[:30], url))
    except Exception as e:
        not_scrapeable.append(('ERROR', str(e)[:30], url))
    time.sleep(0.5)

print(f"\n✅ SCRAPEABLE ({len(scrapeable)}):")
for status, url in scrapeable:
    print(f"  {status}: {url}")

print(f"\n❌ NOT SCRAPEABLE ({len(not_scrapeable)}):")
for status, reason, url in not_scrapeable:
    print(f"  {status} ({reason}): {url}")